"""Tool 7 — the output surface, and the only tool that calls a model.

Two things happen here and the order matters: the object is **assembled from
structured arguments and validated**, and only then is the prose written, from
the assembled object.

### Why the drafting call cannot see the evidence

This tool is given only what the agent has already determined — heading, slab,
differential, citations, and its own short notes. It never receives `evidence`:
not the Gazette entry text, not the advance-ruling excerpt, nothing a document
supplied.

That is a defence, not an optimisation. Composition is the last step to touch
the output, so it is the most valuable place for injected text to reach; cutting
it off from every untrusted channel means an instruction hidden in a document
has no path to the final wording. It also means the justification cannot cite
what the structured determination does not contain.

The cost is real and stated rather than hidden: the prose is thinner than it
would be with the source text in front of it. Whether that costs anything
measurable is a week-5 question, and if it does it goes in FAILURES.md as a
defence with a price.

### Validation runs twice

Once before drafting, because a model call to justify an object that is about
to be rejected is a call spent on nothing — and under chaos that is a real share
of the bill. Once after, because the prose is new text and DESIGN.md §5(6)
applies to it: an object that is correct in every field but whose justification
asserts an abolished rate as current has failed the domain safety property.

### Without a key

Falls back to a deterministic template and labels the result
`justification_source: "template"`. Every downstream number stays computable and
no run reports templated prose as model-written.
"""

from __future__ import annotations

from datetime import date

from agent.contract import ToolResult
from agent.llm import Message, Model, ModelError
from agent.opinion import Citation, Opinion, validate
from agent.registry import ToolSpec

_MODEL: Model | None = None
_MODEL_RESOLVED = False

_DRAFT_SYSTEM = """You write one paragraph of justification for an Indian GST \
rate opinion on a single invoice line.

You are given only a structured determination that tools reading the \
hash-pinned Gazette notifications have already made. Your job is to state it in \
prose. You are NOT deciding the rate.

Rules:
- Use only the facts in the determination. Introduce no heading, rate, \
notification or figure that is not there.
- Cite the notification and schedule you are given.
- If a rate is described as abolished, say so in the past tense. Never state an \
abolished rate as the applicable one.
- Three to five sentences. No preamble, no headings, no bullet points.
"""


def set_model(model: Model | None) -> None:
    """Inject the drafting model. Tests use this to pin the prose."""
    global _MODEL, _MODEL_RESOLVED
    _MODEL = model
    _MODEL_RESOLVED = True


def _model() -> Model | None:
    global _MODEL, _MODEL_RESOLVED
    if _MODEL_RESOLVED:
        return _MODEL
    _MODEL_RESOLVED = True
    try:
        from agent.llm import build

        # 512 tokens: this call writes three to five sentences from a
        # determination that is already made. Reasoning stays off for the same
        # reason — there is nothing left to decide.
        candidate = build(max_tokens=512)
        candidate._ensure_client()  # surface a missing key now, not mid-suite
        _MODEL = candidate
    except Exception:  # noqa: BLE001 — absence of a key is a valid state
        _MODEL = None
    return _MODEL


def _template(op: Opinion, declared_rate: str | None) -> str:
    """The keyless fallback. Deterministic, and labelled as such."""
    if op.terminal == "out_of_scope":
        return (
            f"The line describes goods outside the scope of GST ({op.reason}). "
            "Alcoholic liquor for human consumption is excluded by Article "
            "366(12A) and taxed under state excise, so no GST slab applies and "
            "no differential arises."
        )
    if op.terminal == "unanswerable":
        return (
            f"The line cannot be assessed on the information given ({op.reason}). "
            "The description does not determine a single tariff heading and "
            "rate, so no opinion is offered rather than a rate being assumed."
        )
    cite = op.citations[0] if op.citations else None
    where = (
        f"{cite.notification}, Schedule {cite.schedule}, heading {cite.heading}"
        if cite
        else "the archived notification"
    )
    tail = ""
    if declared_rate is not None:
        tail = (
            f" The supplier declared {declared_rate}%, which matches, so the "
            "differential is nil."
            if op.declared_correct
            else (
                f" The supplier declared {declared_rate}%, so the line is "
                f"misdeclared and the differential is Rs {op.differential_inr}."
            )
        )
    return (
        f"Heading {op.hsn4} applies to these goods, and it is rated at "
        f"{op.slab}% per {where} as in force on {op.invoice_date}.{tail}"
    )


def draft_opinion(
    terminal: str,
    invoice_date: str,
    line_id: str = "",
    hsn4: str | None = None,
    slab: str | None = None,
    declared_correct: bool | None = None,
    differential_inr: str | None = None,
    declared_rate: str | None = None,
    reason: str | None = None,
    citations: list | None = None,
    notes: str = "",
) -> ToolResult:
    """Emit and validate the final rate opinion for the line."""
    try:
        when = date.fromisoformat((invoice_date or "").strip()[:10])
    except ValueError:
        return ToolResult.err(
            "bad_argument", f"invoice_date {invoice_date!r} is not yyyy-mm-dd"
        )

    cites = [
        Citation(
            notification=str(c.get("notification", "")).strip(),
            schedule=str(c.get("schedule", "")).strip(),
            heading=str(c.get("heading", "")).strip(),
        )
        for c in (citations or [])
        if isinstance(c, dict)
    ]

    op = Opinion(
        terminal=str(terminal).strip(),
        line_id=str(line_id or ""),
        invoice_date=when.isoformat(),
        hsn4=str(hsn4).strip() if hsn4 else None,
        slab=str(slab).strip().rstrip("%") if slab not in (None, "") else None,
        answerable=str(terminal).strip() == "opinion",
        declared_correct=declared_correct,
        differential_inr=str(differential_inr) if differential_inr is not None else None,
        reason=str(reason).strip() if reason else None,
        citations=cites,
    )

    if problems := validate(op, invoice_date=when):
        return ToolResult.err(
            "malformed",
            "; ".join(problems),
            data={"problems": problems, "draft": op.to_json()},
        )

    source = "template"
    model = _model()
    if model is not None:
        determination = "\n".join(
            f"{k}: {v}"
            for k, v in op.to_json().items()
            if v not in (None, "", [], {}) and k != "justification"
        )
        if declared_rate:
            determination += f"\ndeclared_rate: {declared_rate}"
        if notes:
            determination += f"\nagent_notes: {notes[:600]}"
        completion = model.complete(
            _DRAFT_SYSTEM,
            [Message(role="user", content=f"Determination:\n{determination}")],
        )
        if completion.ok and completion.text.strip():
            op.justification = completion.text.strip()
            source = "model"

    if source == "template":
        op.justification = _template(op, declared_rate)

    if post := validate(op, invoice_date=when):
        return ToolResult.err(
            "malformed",
            "the justification failed validation after drafting: " + "; ".join(post),
            data={
                "problems": post,
                "draft": op.to_json(),
                "justification_source": source,
            },
        )

    return ToolResult.ok_(
        {
            "opinion": op.to_json(),
            "terminal": op.terminal,
            "justification_source": source,
            "validated": True,
        }
    )


SPEC = ToolSpec(
    name="draft_opinion",
    description=(
        "Emit the final rate opinion for the line and finish the run. Call this "
        "exactly once, when you have determined the outcome. terminal is "
        "'opinion' (a slab was determined), 'out_of_scope' (no GST applies) or "
        "'unanswerable' (the line does not determine a rate — give a reason "
        "code). The object is validated: an invalid one comes back as malformed "
        "with the problems listed, and you should correct it and call again. A "
        "slab that did not exist on the invoice date is rejected."
    ),
    parameters={
        "type": "object",
        "properties": {
            "terminal": {
                "type": "string",
                "description": "The outcome for this line.",
                "enum": ["opinion", "out_of_scope", "unanswerable"],
            },
            "invoice_date": {
                "type": "string",
                "description": "The invoice date, yyyy-mm-dd.",
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
            "line_id": {"type": "string", "description": "The line id from the input."},
            "hsn4": {
                "type": "string",
                "description": "The 4-digit heading. Required for 'opinion'.",
                "pattern": r"^\d{4}$",
            },
            "slab": {
                "type": "string",
                "description": (
                    "The rate as a percentage string, e.g. '18'. Required for "
                    "'opinion'. Must have existed on the invoice date."
                ),
            },
            "declared_correct": {
                "type": "boolean",
                "description": "Whether the rate the supplier charged was right.",
            },
            "differential_inr": {
                "type": "string",
                "description": "The differential from compute_liability, e.g. '15000.00'.",
            },
            "declared_rate": {
                "type": "string",
                "description": "The rate the supplier charged, for the narrative.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "Required for 'out_of_scope' and 'unanswerable'. One of: "
                    "alcoholic-liquor, no-product-kind, model-number-only, "
                    "rate-fact-absent, packaging-indeterminate, "
                    "multi-good-no-dominant, date-outside-archive, "
                    "heading-not-in-schedule."
                ),
            },
            "citations": {
                "type": "array",
                "description": (
                    "Schedule entries relied on. Required for 'opinion'. Each is "
                    "{notification, schedule, heading}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "notification": {"type": "string"},
                        "schedule": {"type": "string"},
                        "heading": {"type": "string"},
                    },
                    "required": ["notification", "schedule", "heading"],
                    "additionalProperties": False,
                },
                "maxItems": 8,
            },
            "notes": {
                "type": "string",
                "description": (
                    "Your own short summary of how you reached this, in your "
                    "words. Used to compose the justification. Do not paste "
                    "document text here."
                ),
                "maxLength": 2000,
            },
        },
        "required": ["terminal", "invoice_date"],
        "additionalProperties": False,
    },
    handler=draft_opinion,
    stage="draft",
    returns_evidence=False,
    # Not pure: the drafting call is a model call, so two invocations can
    # produce different prose. Idempotent only because the cache makes it so,
    # which is exactly the residual risk IdempotencyCache documents.
    pure=False,
)
