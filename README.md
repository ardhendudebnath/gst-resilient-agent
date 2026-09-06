# Indian GST rate-opinion agent — a reliability study

A seven-tool agent that audits a line from an Indian GST invoice: **was the rate
the supplier charged the rate that was actually in force on the date of that
invoice, and if not, what is the exposure?** Plus the chaos harness built to
break it, and the taxonomy of what breaking it turns up.

> **Status: week 2 of 7. There are no reliability results yet, and the tables
> below are empty because nothing has been measured.** They will stay empty
> until it has. The agent runs end to end and the tool layer is tested; the
> chaos harness, the task suite and the failure taxonomy are not built. Nothing
> in this README is claimed that has not been run.

The companion benchmark is [gst-eval-harness](https://github.com/ardhendudebnath/gst-eval-harness),
which measures whether a model can name the current GST rate at all. This
repository is the other half: what happens when you give a model tools, and then
start breaking them.

---

## Why the date is the whole problem

India restructured its GST slabs twice in seventeen months:

| Date | Notification | Effect |
|---|---|---|
| 2025-09-22 | 9/2025-CT(R) | Supersedes 1/2017. **The 12% slab ceases to exist.** |
| 2026-02-01 | 19/2025-CT(R) | Omits Schedule VII. **The 28% slab ceases to exist**; tobacco and pan masala move to 40%, biris to 18%. |

So heading 2402 (cigarettes) attracts **28% on an invoice dated 12 November
2025** and **40% on one dated 1 March 2026**, and both are correct. An audit that
scores the November line against today's table reports a compliant supplier as
having short-paid by twelve points.

That is not a hypothetical. The companion benchmark measured an open-weight
frontier-class model asserting an **abolished** rate as current in a mean 18.3%
of single-call responses, and reproducing its own answer on only 62.5% of
examples. The rate table is not in the weights, and asking harder does not put
it there.

## Why this needs an agent rather than one good prompt

Four decisions that cannot be made until a previous call has returned:

1. **Is there anything to classify?** Alcoholic liquor is outside GST by
   constitutional exclusion — no slab exists, and the correct behaviour is to
   stop before any lookup.
2. **Which heading?** Zero, one and several candidates are three different paths.
3. **Did the lookup resolve?** `lookup_schedule` **refuses to guess**. 7418
   splits on whether an article is a household article of copper, 8711 on engine
   capacity, 2202 on added sugar. The agent must settle the condition or decline.
4. **Which notification was in force?** See the table above. An invoice dated
   before 2025-09-22 falls under a schedule this repository does not archive, and
   the honest terminal state is a refusal.

`docs/DESIGN.md` §2 states the falsifier: a one-shot baseline over the same suite,
scored by the same scorer, is scheduled for week 3. If a single call with the
notification text pasted in scores within noise of the agent, that goes in this
README as a finding rather than in a drawer.

---

## Reliability under adversarial conditions

Not yet measured. The chaos harness exists and is tested; the **task suite that
turns it into a number does not**, so there is nothing to put in this table yet.
It is the deliverable of weeks 4–7 and will report success rate at each
injection rate, before and after fixes.

| Condition | Baseline | After fixes | Δ |
|---|---|---|---|
| Clean (no chaos) | — | — | — |
| 10% failure injection | — | — | — |
| 25% failure injection | — | — | — |
| 50% failure injection | — | — | — |
| Injection attacks | — | — | — |

## The chaos harness

Eleven failure modes, injected between the loop and its tools at a configured
rate, reproducibly.

| mode | phase | what it does |
|---|---|---|
| `timeout` | before | the tool hangs, then errors |
| `rate_limited` | before | the tool returns 429 |
| `slow` | around | correct results, 20× later |
| `duplicate` | around | the response arrives twice |
| `empty` | after | valid, well-formed, and says nothing |
| `partial` | after | half the fields, silently |
| `wrong_type` | after | a string where a number belonged |
| `malformed_json` | after | output that does not match the tool's contract |
| `contradictory` | after | a fact incompatible with one already given |
| `stale` | after | correct-looking but outdated |
| `injection` | after | document content carrying instructions |

```bash
python -m agent --demo --chaos 0.25 --seed 7
python -m agent --demo --chaos 0.5 --chaos-modes stale,contradictory --seed 3
```

Three properties everything is built around:

- **The agent cannot tell.** `ChaosDispatcher` has the same `invoke` signature
  as `Registry`, and the loop strips the chaos label before a result enters the
  message history. A test asserts the string never reaches the model.
- **Every fabricated result is labelled twice** — on the result and as its own
  trace event — so no injected failure can be mistaken for an organic one when
  the taxonomy is written. This matters more than it sounds: the 550B endpoint
  produces real 503s, and organic and injected failures must stay separable.
- **A seed reproduces a run.** Measured over 120 seeds, configured 10/25/50%
  yields effective 8.2/24.9/49.3%.

Two modes are worth reading the implementation of. **`stale` fabricates
nothing** — it re-runs the real tool with the invoice date moved to the day
before Notification 19/2025 took effect, so the agent gets a genuinely correct
answer to a question nobody asked: 28% for a heading that moved to 40%, read out
of the actual Gazette. **`duplicate` should be harmless**, and proving it is the
point: every tool is a pure function of its arguments and `compute_liability`
takes the running total as an argument rather than accumulating it, so the
double-counting failure the brief predicts cannot occur. Each run reports
`idempotency_held` rather than assuming it.

Effective rate is reported next to the configured rate, because they diverge: a
call the idempotency cache will serve is never perturbed, since it never reaches
the thing that would fail. **Deduplication therefore reduces an agent's exposure
to chaos**, which is a real effect that would otherwise be invisible.

## Retrieval: keyword against embeddings

Candidate recall is the ceiling on the whole workflow — a gold heading the
retriever never proposes cannot be reached however well the agent reasons — so
it is measured rather than assumed. Recall@k over the 28 golden rows, retrieval
channel only (the advocacy channel is excluded, so these are lower than the
tool's end-to-end candidate coverage):

| backend | R@1 | R@3 | R@5 | R@10 | p50 latency |
|---|---:|---:|---:|---:|---:|
| keyword (bag-of-words) | 32.1% | 42.9% | 46.4% | 60.7% | **9 ms** |
| **semantic** (`nemotron-3-embed-1b`) | **46.4%** | **64.3%** | **75.0%** | **89.3%** | 715 ms |
| hybrid (reciprocal rank fusion) | 32.1% | 57.1% | 64.3% | 82.1% | 708 ms |

Rows where the gold heading is never proposed at all fall from **11/28 to 3/28**.

**Hybrid is worse than pure semantic**, which is the opposite of the usual
assumption and the more interesting number. RRF weights both backends equally,
so a keyword list with 32% R@1 drags down a semantic list with 46%. Fusion helps
when the two are comparable; here one is simply better, and blending it with a
weaker signal costs 11 points of R@5.

**The default stays keyword.** Switching the retriever outright would move the
baseline the chaos results are measured against, quietly turning the before/after
table into "keyword versus embeddings" instead of "no defences versus defences".
The backend is switchable, recorded on every call as `retrieval_mode`, and gets
its own row rather than contaminating someone else's.

```bash
python -m agent --demo --retrieval semantic
```

### Three things deliberately not used

- **No vector database.** The rated schedule holds 961 entries; at 2048
  dimensions that is 7.9 MB, one contiguous read, and a full scan is 2 million
  multiply-adds. pgvector earns its place at 10⁵–10⁷ vectors, not 10³, and a
  Postgres dependency would break `make test` on a fresh clone. `VectorIndex` is
  the seam: swap it when the corpus becomes the advance-ruling archive.
- **No PyMuPDF.** It is AGPL-3.0 and this repository is MIT. `pypdf` (BSD)
  already extracts all 961 entries from the hash-pinned PDFs, under test.
- **No fixed-size chunking.** The Gazette is not prose — every row is a serial
  number, a heading, a description and a rate — and DESIGN §5 requires each
  citation to resolve to a specific entry. A sliding token window would cut
  entries in half and make that impossible. One entry, one vector.

Embeddings are cached to disk keyed by model *and* corpus digest, so a
re-vendored Gazette or a model swap invalidates them rather than answering from
vectors built against a different document.

## The failure taxonomy

`FAILURES.md` does not exist yet — it needs the task suite, so that a class can
carry a frequency rather than an anecdote. Target is 12–20 documented classes,
each with its trigger, frequency, root cause, fix and residual risk.

## Security testing

The OWASP LLM Top 10 coverage table is week 5. What exists today is the
structure the tests will need:

- **Trusted and untrusted data are separate fields.** `ToolResult.data` is
  computed by this repository; `ToolResult.evidence` is verbatim document text
  with its provenance. See `agent/contract.py`.
- **Retrieved content never reaches a system-prompt position.** Structural, not
  switchable — pinned by a test.
- **Defences are switchable and default off**, so the baseline is honest. See
  `agent/policy.py` and `docs/DESIGN.md` §6.

**The corpus is adversarial before anything is injected into it.** Advance
rulings carry the applicant's own contention, argued in the first person and
frequently the contention the authority rejected:

> "The applicant is of the opinion that correct classification of such Quartz
> Slabs is under HSN 6810 … attracting GST @ 18%."

One measurement already exists, and it complicates the week-5 plan. Across the
28 golden rows, the gold heading appears in the document's own advocacy **20/28**
times, but in independent Gazette retrieval only **13/28**. An agent that simply
defers to what the document argues for scores *better* than one reasoning from
the notification — so "did the agent resist the injected instruction" cannot be
measured by accuracy alone, and needs its own metric.

---

## Architecture

```
              invoice line (JSON)
                     │
              ┌──────▼───────┐
              │  agent/loop  │  hand-rolled while loop, bounded,
              │              │  one JSON action per turn
              └──────┬───────┘
                     │  ToolCall
            ┌────────▼─────────┐
            │  chaos/          │  ← week 4: injects 11 failure modes,
            │  middleware      │    same signature, agent cannot tell
            └────────┬─────────┘
            ┌────────▼─────────┐
            │  agent/registry  │  validates args, dedupes on
            │  .invoke         │  (tool, args), traces, caps per tool
            └────────┬─────────┘
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
  screen_scope  lookup_schedule   …7 tools…   → ToolResult{data, evidence}
                     │
                     ▼
        data/reference/primary/*.pdf   SHA-256 verified on every read
```

Seven tools, in workflow order:

| # | Tool | The branch it forces |
|---|---|---|
| 1 | `screen_scope` | terminal stop on 2 of 3 verdicts |
| 2 | `propose_headings` | 0 / 1 / many are three paths |
| 3 | `lookup_schedule` | resolved / **ambiguous** / chapter-only / absent |
| 4 | `check_conditions` | not determinable ⇒ `unanswerable` |
| 5 | `rate_history` | pre-2025-09-22 ⇒ outside archive ⇒ refusal |
| 6 | `compute_liability` | exact decimal arithmetic |
| 7 | `draft_opinion` | schema-validated output surface; every path ends here |

`lookup_schedule` is the one exposed as an MCP server (`mcp_gazette/`, not built
yet). It is the right one to expose: four genuinely different return types, real
error responses, and it is independently useful — "look an Indian tariff heading
up in the archived Gazette and get a citable rate" is a thing other people want.

### Provenance

Every rate traces to a hash-pinned Gazette notification in
`data/reference/primary/`, verified against `MANIFEST.json` on every read. A tool
that answers from a document it cannot prove is the pinned one returns
`source_mismatch` rather than a rate — a wrong answer carrying a citation is the
most convincing kind.

---

## Reproducing

```bash
pip install -e '.[gazette,dev]'
cp .env.example .env    # add NVIDIA_API_KEY
```

That is the whole install. The default model path is NVIDIA's API catalog,
reached with one POST over stdlib `urllib`, so no provider SDK is needed —
`pypdf` for the Gazette PDFs is the only runtime dependency.

The default model is **`nvidia/nemotron-3-ultra-550b-a55b`**, open weights at
frontier scale. The agent's job is multi-step judgement over tool output —
reading a schedule entry and deciding whether it describes these goods — which
is where the gap between 120B and 550B actually shows.

**The trade that makes, stated rather than buried.** Project 01 deliberately did
*not* use 550B for its open-weight row, on the grounds that nobody self-hosts
550B. Making it the default here costs the self-hostability demonstration and
the clean bridge to Project 03. Both are recoverable — the wire format is
identical, so

```bash
python -m agent --demo --model nvidia/nemotron-3-super-120b-a12b
```

reproduces any run against the self-hostable sibling, which has a published NIM
container and runs on one node. Project 03 should use that id, and any published
comparison should report both rows rather than only the stronger one.

Run the worked example from `docs/DESIGN.md` §1:

```bash
python -m agent --demo
```

Reasoning is **off** by default. Nemotron emits its chain into a separate field
that still bills as output tokens, and a 12-turn loop against the 60k budget in
`docs/DESIGN.md` §10 would exhaust itself before reaching an opinion. Turn it on
with `--thinking` and raise `--max-tokens-budget` to match; either way it is
recorded on every completion, so a result can never imply reasoning was on when
it was not.

The test suite needs no API key and no network:

```bash
python -m pytest tests -q
```

Verify the archived notifications against their hashes:

```bash
python -c "from agent.gazette import verify_sources; print(verify_sources())"
```

---

## What still doesn't work

Named specifically, because this section is a feature.

- **The task suite, the MCP server and the trace viewer do not exist.**
  `suite/` and `mcp_gazette/` are empty packages. Without the suite the chaos
  harness produces single runs, not rates, so there is still no headline number.
- **The injection payloads have detectors but no measured compliance.** Five
  attack shapes are implemented with a scoring function each — the point being
  that "the agent seemed to ignore it" is not a measurement — but nothing has
  been run at scale, so no compliance rate is claimed.
- **`malformed_json` cannot corrupt a wire format**, because there isn't one:
  results travel as Python objects. What it does instead is deliver a result
  whose structure is not the one the tool's contract promised, with `ok` true.
  That is the failure that reaches the agent; the transport-level one is out of
  reach here and is not claimed to be tested.
- **`check_pass` and `recovery_policies` are declared and unimplemented.**
  Switching either on raises rather than silently doing nothing.
- **`check_conditions` covers five headings** (7418, 8711, 2202, 9608, 2403).
  Any other ambiguous heading returns `not_covered`, and the correct response is
  a refusal. There is no general condition-resolver and there could not be.
- **Candidate recall caps the suite**, though less tightly than it did.
  Retrieval alone reaches the gold heading for 60.7% of rows on keyword and
  89.3% on semantic (R@10); with the advocacy channel added, keyword covers
  roughly 82%. Rows the retriever never proposes cannot be answered correctly
  however good the reasoning is.
- **Semantic retrieval does not fix the worked example.** Gold heading 6810
  reads "Articles of cement, of concrete or of artificial stone"; embeddings
  rank the *mineral* headings (2506, 2505, 2504) above it for "quartz slabs …
  polyester resin", because the description is about the material and the
  correct heading is about the article made from it. That distinction is the
  classification problem itself, not a retrieval problem. On the live runs the
  agent verified the declared heading 6802 instead, found it genuinely ambiguous
  (5% against 18%), found no condition rule encoded for it, and **refused** —
  the designed behaviour on the information available, and still the wrong
  answer.
- **`--demo` does not currently produce the right answer**, for the reason
  above. It is left that way rather than tuned: fitting the retriever to one
  example is exactly what the week-3 suite exists to prevent, and a demo that
  passes because it was hand-fitted is worth nothing.
- **The 550B endpoint returns `503 Service temporarily overloaded` often.**
  Measured, not impressionistic: an eight-step run met three, and two runs since
  met two each. The loop retries transient provider failures with jittered
  backoff and does **not** charge them to the iteration budget — an earlier
  version did, which would have ended longer lines in `budget_exhausted` and
  recorded an infrastructure failure as the agent giving up. Every run reports
  `model_retries` for exactly this reason: a suite quietly absorbing 503s is
  measuring the endpoint's mood rather than the agent. This needs watching
  before the chaos suite runs hundreds of tasks, and an organic 503 must never
  be confusable with an injected one.
- **Tool 7 falls back to a deterministic template when its model call fails**,
  and says so via `justification_source`. It retries transient failures first,
  but a run can still end with templated prose. The determination is unaffected
  — the numbers come from tools — but anything scoring the justification has to
  exclude those runs.
- **All 28 golden rows are `gazette-derived`** — slab read out of the pinned
  notification, heading from each authority's operative ruling, **no human has
  confirmed any of them.** Every accuracy figure inherits that. A before/after
  delta is still meaningful, since the same imperfect reference scores both
  sides, but an absolute success rate from this suite is agreement with a
  document lookup, not with a human.
- **28 rows is too few.** `docs/DESIGN.md` §9 covers what the suite does about
  it; constructed scenarios are marked `synthetic` and scored in their own
  column, never averaged in.

## Licence

MIT. The Gazette notifications in `data/reference/primary/` are Government of
India publications, included unmodified for verifiability.
