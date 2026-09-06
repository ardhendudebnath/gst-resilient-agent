"""NVIDIA embeddings, over stdlib urllib, with a disk cache.

`nemotron-3-embed-1b`, 2048 dimensions, served on the same catalog as the chat
model, so a key that works for one works for the other and no SDK is needed.

**Asymmetric.** The model takes an `input_type` of `passage` or `query` and
embeds them differently: a tariff entry is a passage, a goods description is a
query. Sending the wrong one is not an error and produces quietly worse
retrieval, which is the kind of bug that shows up as "embeddings did not help
much" rather than as a failure. It is a required argument here for that reason.

**Cached to disk, keyed by the corpus.** Embedding 961 schedule entries is one
API round trip per batch and the schedules are hash-pinned, so the result is
reused until the corpus changes. The cache key includes the corpus digest, so a
re-vendored Gazette invalidates it rather than silently answering from vectors
built against a different document.

Vectors are stored as raw float32 via `array`, not JSON: 961 x 2048 floats is
7.9 MB packed and roughly 40 MB as text, and the packed form loads in one read.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from agent.config import REPO_ROOT
from agent.llm import backoff_delay, is_transient

EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

#: Pinned exactly. Hosted embedding ids get retired the same way chat ids do —
#: `llama-3.2-nv-embedqa-1b-v2` already returns 410 Gone — and an index built
#: with one model is meaningless to another, so the model name is part of the
#: cache key rather than a footnote.
DEFAULT_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
EMBED_DIM = 2048

#: The catalog rejects oversized batches, and a 961-entry corpus is 16 requests
#: at this size. Small enough to retry cheaply when one fails.
BATCH_SIZE = 64

MAX_EMBED_RETRIES = 4

CACHE_DIR = Path(os.environ.get("EMBED_CACHE_DIR", "") or REPO_ROOT / "data" / "cache" / "embeddings")


class EmbeddingError(RuntimeError):
    """The embedding service could not be reached, or is not configured."""


def embed_model() -> str:
    return os.environ.get("EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODEL


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def corpus_digest(texts: Sequence[str]) -> str:
    """A digest over the exact strings embedded, in order.

    Order matters: the index maps row N to entry N, so a reordered corpus is a
    different index even when the set of strings is identical.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class EmbeddingClient:
    """One POST per batch. No SDK, so this adds no runtime dependency."""

    def __init__(
        self,
        model: str | None = None,
        *,
        batch_size: int = BATCH_SIZE,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or embed_model()
        self.batch_size = batch_size
        self._timeout = timeout
        self._key = os.environ.get("NVIDIA_API_KEY", "").strip()

    def available(self) -> bool:
        return bool(self._key)

    def _post(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        payload = {
            "input": list(texts),
            "model": self.model,
            "input_type": input_type,
            "encoding_format": "float",
            # The schedules run to a couple of hundred characters an entry, well
            # inside the window, but a truncation policy has to be stated or the
            # service picks one.
            "truncate": "END",
        }
        req = urllib.request.Request(
            EMBED_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        rows = sorted(body.get("data") or [], key=lambda d: d.get("index", 0))
        return [r["embedding"] for r in rows]

    def embed(self, texts: Sequence[str], *, input_type: str) -> list[list[float]]:
        """Embed `texts`. `input_type` is 'passage' (corpus) or 'query'."""
        if input_type not in ("passage", "query"):
            raise ValueError(
                f"input_type must be 'passage' or 'query', got {input_type!r}. "
                "The model embeds them differently and the wrong one degrades "
                "retrieval silently."
            )
        if not self._key:
            raise EmbeddingError(
                "NVIDIA_API_KEY is not set, so no embeddings can be built. "
                "Keyword retrieval still works; see agent/retrieval.py."
            )
        if not texts:
            return []

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            for attempt in range(1, MAX_EMBED_RETRIES + 2):
                try:
                    out.extend(self._post(batch, input_type))
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read()[:240].decode("utf-8", "replace")
                    error = f"http_{exc.code}: {detail}"
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                if not is_transient(error) or attempt > MAX_EMBED_RETRIES:
                    raise EmbeddingError(
                        f"embedding batch {start // self.batch_size} failed: {error}"
                    )
                time.sleep(backoff_delay(attempt))
        return out


# --------------------------------------------------------------------------
# Vector maths. Stdlib, because 961 vectors do not need a library.
# --------------------------------------------------------------------------


def normalise(vector: Sequence[float]) -> array:
    """Unit-length float32. Normalising once at build time turns every later
    cosine similarity into a plain dot product."""
    total = 0.0
    for v in vector:
        total += v * v
    norm = total ** 0.5 or 1.0
    return array("f", [v / norm for v in vector])


def dot(a: array, b: array) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(slots=True)
class VectorIndex:
    """Row-major float32 vectors with parallel ids. Brute-force search.

    Brute force is the right algorithm here and saying why matters, because the
    obvious instinct is to reach for a vector database. The rated schedule holds
    961 entries; at 2048 dimensions that is 7.9 MB, one contiguous read, and a
    full scan costs 2 million multiply-adds. An ANN index and a Postgres
    dependency would buy nothing measurable and would cost `make test` on a
    fresh clone.

    The seam is here rather than absent: swap this class for a pgvector-backed
    one when the corpus is the advance-ruling archive — thousands of chunks
    rather than hundreds of entries — and the callers do not change.
    """

    model: str
    dim: int
    ids: list[str]
    rows: list[array]
    corpus_sha: str = ""

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: Sequence[float], limit: int = 8) -> list[tuple[str, float]]:
        q = normalise(query)
        scored = [(self.ids[i], dot(q, row)) for i, row in enumerate(self.rows)]
        scored.sort(key=lambda r: -r[1])
        return scored[:limit]

    # -- persistence -----------------------------------------------------

    def save(self, stem: Path) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        flat = array("f")
        for row in self.rows:
            flat.extend(row)
        stem.with_suffix(".vec").write_bytes(flat.tobytes())
        stem.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "dim": self.dim,
                    "ids": self.ids,
                    "corpus_sha": self.corpus_sha,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, stem: Path) -> "VectorIndex | None":
        meta_path = stem.with_suffix(".meta.json")
        vec_path = stem.with_suffix(".vec")
        if not (meta_path.is_file() and vec_path.is_file()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        flat = array("f")
        flat.frombytes(vec_path.read_bytes())
        dim = int(meta["dim"])
        ids = list(meta["ids"])
        if len(flat) != dim * len(ids):
            # A truncated cache is worse than none: it would answer with
            # vectors misaligned to their ids, which reads as a bad model.
            return None
        rows = [array("f", flat[i * dim : (i + 1) * dim]) for i in range(len(ids))]
        return cls(
            model=meta["model"],
            dim=dim,
            ids=ids,
            rows=rows,
            corpus_sha=meta.get("corpus_sha", ""),
        )


def index_path(name: str, model: str | None = None) -> Path:
    return CACHE_DIR / f"{name}-{_slug(model or embed_model())}"


def build_index(
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    name: str,
    client: EmbeddingClient | None = None,
    rebuild: bool = False,
) -> VectorIndex:
    """Load a cached index, or build and cache one.

    The cache is invalidated by the corpus digest *and* the model name, so
    neither a re-vendored Gazette nor a model swap can be answered from stale
    vectors — the failure that looks like a mysterious drop in retrieval
    quality and takes a day to find.
    """
    if len(ids) != len(texts):
        raise ValueError(f"{len(ids)} ids against {len(texts)} texts")

    client = client or EmbeddingClient()
    stem = index_path(name, client.model)
    digest = corpus_digest(texts)

    if not rebuild:
        cached = VectorIndex.load(stem)
        if cached is not None and cached.corpus_sha == digest and cached.model == client.model:
            return cached

    vectors = client.embed(list(texts), input_type="passage")
    index = VectorIndex(
        model=client.model,
        dim=len(vectors[0]) if vectors else EMBED_DIM,
        ids=list(ids),
        rows=[normalise(v) for v in vectors],
        corpus_sha=digest,
    )
    index.save(stem)
    return index


def embed_query(text: str, client: EmbeddingClient | None = None) -> list[float]:
    """One query vector. Separate from `embed` so the `input_type` cannot be
    got wrong at a call site."""
    client = client or EmbeddingClient()
    vectors = client.embed([text], input_type="query")
    if not vectors:
        raise EmbeddingError("the embedding service returned no vector")
    return vectors[0]
