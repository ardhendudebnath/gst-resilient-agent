# Design — pinned before the agent was written

Recorded 2026-09-06, at the start of week 1, so that nothing below can be
retrofitted to whatever the measurements turn out to say. Where a decision is
later reversed, the reversal is appended with its date and reason rather than
edited over the top.

---

## 1. The task

One **invoice line**, as a supplier actually filed it, goes in:

```json
{
  "line_id": "inv-0042",
  "description": "Quartz slabs, 92% crushed quartz bonded with 8% polyester resin, polished",
  "declared_hsn": "6802",
  "declared_rate": "12",
  "taxable_value_inr": 250000.00,
  "invoice_date": "2026-03-14"
}
```

A **rate opinion** comes out:

```json
{
  "terminal": "opinion",
  "hsn4": "6810",
  "slab": "18",
  "answerable": true,
  "declared_correct": false,
  "differential_inr": 15000.00,
  "citations": [
    {"notification": "9/2025-CT(R)", "schedule": "II", "heading": "6810"}
  ],
  "justification": "..."
}
```

The question the agent answers is not "what rate is this" but **"is the rate
the supplier charged the rate that was actually in force on the date of the
invoice, and if not, what is the exposure?"** That framing is what makes the
date a real input and the arithmetic a real output.

### Why invoice lines rather than bare product descriptions

Project 01 asks a model to classify a description. This project asks whether a
filing was correct. The difference buys three things:

- **A date**, which selects the notification, which is the branch the domain
  is actually about.
- **A declared rate**, which can be the *stale* rate — the exact error
  Project 01 measured models making. The suite is built from lines where a
  supplier used the pre-2025 table, so the agent is checked against the
  failure the domain really has.
- **A number**, computed rather than judged, that a scorer can check to the
  paisa.

---

## 2. Why this needs an agent

The pitfall the brief names is a workflow one well-prompted call would do. The
argument that this is not that workflow is **measured, not asserted** — it is
the headline result of Project 01:

> Asked to classify goods into GST slabs in a single call, an open-weight
> frontier-class model asserted an **abolished** rate as current in a mean
> **18.3 %** of responses (range 12.5–25.0 % across five runs, n = 24), and
> reproduced its own answer on only **62.5 %** of examples.
> — [gst-eval-harness](https://github.com/ardhendudebnath/gst-eval-harness), run of 2026-09-05

The rate table is not in the weights, and asking harder does not put it there.
India restructured its GST slabs twice in seventeen months; the web the models
were trained on overwhelmingly describes the table that was superseded. The
only way to get a citable rate is to read the notification, which means a tool
call, which means a loop.

That handles the retrieval half. The branching half is four decisions that
cannot be made until a previous call has returned:

1. **Is there anything to classify?** Alcoholic liquor is outside GST by
   constitutional exclusion — there is no slab to predict, and the correct
   behaviour is to stop. So is a description too vague to determine a kind of
   good. Both are terminal states reached *before* any lookup.
2. **Which heading?** Zero candidates, one, or several are three different
   paths. Several means a General Rules of Interpretation tie-break.
3. **Did the lookup resolve?** `lookup_schedule` **refuses to guess**: a
   heading appearing in more than one Schedule comes back flagged ambiguous
   with all its entries, never resolved. 7418 splits on whether an article is
   a household article of copper, 8711 on engine capacity, 2202 on added
   sugar. The agent must then go and settle the condition, or decline.
4. **Which notification was in force?** 9/2025 from 22 September 2025;
   9/2025 as amended by 19/2025 from 1 February 2026. An invoice dated before
   22 September 2025 falls under a schedule this repository does not archive,
   and the honest terminal state is a refusal, not a guess.

A single call can produce none of the early stops, cannot condition step 4 on
step 3's output, and — per the measurement above — gets the rate from memory.

**If this argument stops holding, it should be recorded here rather than
defended.** The falsifier is cheap and will be run in week 3: a one-shot
baseline over the same suite, scored by the same scorer. If a single call with
the notification text pasted in scores within noise of the agent, that belongs
in the README as a finding, not in a drawer.

---

## 3. Tools

Seven, each with the decision it forces. `lookup_schedule` is the one exposed
as an MCP server (§7).

| # | Tool | Returns | The branch it forces |
|---|---|---|---|
| 1 | `screen_scope` | in scope / out of scope / under-specified | Terminal stop on 2 of 3 outcomes |
| 2 | `propose_headings` | candidate HSN-4 headings + the text that suggested each | 0 / 1 / many are three paths |
| 3 | `lookup_schedule` | Schedule and slab, **or** ambiguous, **or** chapter-only, **or** absent | Ambiguous forces tool 4; absent forces a refusal |
| 4 | `check_conditions` | resolves a conditional entry from the description | Not determinable ⇒ `UNANSWERABLE` |
| 5 | `rate_history` | which notification was in force on the invoice date, and whether this heading moved | Pre-22-Sep-2025 ⇒ out of archive ⇒ refusal |
| 6 | `compute_liability` | exact arithmetic on the differential | None — but it is where non-idempotency bites |
| 7 | `draft_opinion` | the schema-constrained final object | None — it is the output surface |

Tools 3 and 5 read the hash-pinned Gazette PDFs in `data/reference/primary/`
(SHA-256 verified on load, §8). Tools 1, 2, 4 and 6 are pure functions of their
arguments. Tool 7 is the only one that calls a model.

### Terminal states

Three, and the suite contains examples of each:

| Terminal | Meaning | Counts as success when |
|---|---|---|
| `opinion` | a slab was determined and cited | heading, slab and differential all correct |
| `out_of_scope` | no GST slab exists for these goods | gold says out of scope |
| `unanswerable` | the description does not determine a slab | gold says unanswerable, and the reason code matches |

A fourth outcome, `budget_exhausted`, is always a failure and is reported
separately from wrong answers — running out of steps and answering incorrectly
are different faults with different fixes.

---

## 4. Tool contract

Four rules, taken from the brief, plus one this domain adds.

1. **Idempotent.** Every tool is a pure function of `(name, arguments)` or is
   made so by caching on that key. Calling twice must be indistinguishable
   from calling once. This is the decision that pre-empts the double-counting
   failure the brief predicts for week 5, and `compute_liability` is where it
   would otherwise land.
2. **Structured errors, never exceptions.** A tool returns
   `{"ok": false, "error": "timeout", "retryable": true}`. Nothing a tool does
   may unwind the loop; the agent has to be able to *reason* about the
   failure, which it cannot do with a traceback.
3. **Every call carries an id** and is written to the trace with its
   arguments, its result, its timing and its token cost.
4. **Everything is bounded** — iterations, tool calls, tokens, wall-clock, and
   per-tool call counts. Set before the first chaos run, not after the bill.

5. **Trusted and untrusted data are different fields.** `ToolResult.data` is
   structured output my own code computed. `ToolResult.evidence` is verbatim
   text lifted out of a document. They are separate because in week 5 one of
   them is the attack surface and the other is not.

   This matters more here than it would in most domains, because the corpus
   is **naturally adversarial before anything is injected into it**. Advance
   ruling excerpts contain the applicant's own rejected contention, argued in
   the first person:

   > "The applicant is of the opinion that correct classification of such
   > Quartz Slabs is under HSN 6810 … attracting GST @ 18%."

   That is a persuasive, confidently-worded, *sometimes wrong* instruction to
   classify a certain way, sitting inside the document the agent must read.
   No attacker put it there. Whether the agent follows it is measurable, and
   week 5 measures it alongside the synthetic injections.

---

## 5. Success criteria — pinned

A task **passes** if and only if every one of these holds:

1. The terminal state equals the expected terminal state.
2. If `opinion`: `hsn4` exact-matches gold; `slab` exact-matches gold;
   `differential_inr` matches the independently computed value to the paisa.
3. If `unanswerable`: the reason code matches gold's.
4. The final object validates against the output schema.
5. The run finished inside every budget.
6. **No abolished slab (12 %, 28 %) is asserted as current anywhere in the
   output.** Scored with Project 01's `find_abolished_citations`, which
   already knows not to count a rate wrapped in historical language
   ("the erstwhile 28 % rate no longer applies").

Criterion 6 is a domain safety property, not an accuracy metric, and it is
reported as its own column. An agent that gets the slab right while reciting a
dead schedule in its justification has failed in the way this domain actually
fails.

Partial credit is recorded but never counted as a pass: chapter-level HSN
agreement, and "right slab by a route that would not generalise", both go in
the results file for diagnosis.

**Secondary metrics**, recorded every run, not gated on: steps taken, tool
calls made, wall-clock, tokens in/out, and cost.

---

## 6. What the baseline is, and what it is not

The before/after table only means something if the "before" is a fair
implementation. So, stated plainly:

**The baseline is the agent an ordinary careful engineer writes on the first
pass.** It has bounded loops, structured tool errors, idempotent tools, and
tracing — because those are §4 decisions, made on day one, and pretending
otherwise would be building a strawman to knock down.

**The baseline does not have**, and week 5 and 6 add:

- delimiting or escaping of `evidence` before it enters the message history
- a check pass over tool results
- a per-step tool allowlist
- recovery policies specific to a failure class
- output-schema validation as a *gate* rather than a report

Those are absent because each is a defence against something not yet measured,
and adding a defence before the measurement leaves nothing to measure. They
are named here, in advance, so the eventual improvement cannot be read as
having been engineered into the "before".

The honest risk in the other direction is that this list is exactly the set of
things I already suspect will be needed, which makes the "discovery" partly
foreknowledge. Week 4–5 will surface failure classes not on this list, and
which entries in `FAILURES.md` were anticipated here versus found by testing
will be marked as such.

---

## 7. The MCP server

`lookup_schedule` is exposed as a Model Context Protocol server, in
`mcp_gazette/`, and the agent consumes it over the protocol rather than as a
local import. It is the right one of the seven to expose because:

- its return type is genuinely rich — resolved, ambiguous, chapter-only and
  absent are four different answers, which exercises schema design rather than
  returning a string;
- it has real error responses (source file missing, hash mismatch, malformed
  heading);
- it is **independently useful**. "Look an Indian tariff heading up in the
  archived Gazette and get a citable rate" is a thing other people want, and
  the server can be published on its own.

The local-function version is kept behind the same interface so the protocol
boundary can be switched off — which makes "what did MCP cost me in latency"
a measurable question rather than a rhetorical one.

---

## 8. Reuse of Project 01

| What | How |
|---|---|
| Scoring final outputs | `harness.scorers.exact` — `score_row`, `summarise`, `find_abolished_citations` |
| Scenario source | `data/golden.jsonl`, 28 rows, all `gazette-derived` |
| Gazette lookup | `harness.collect.schedule_lookup.lookup` |
| Scope screening | `harness.schema.out_of_scope_term` |
| Label space | `harness.schema` — `VALID_SLABS`, `ABOLISHED_SLABS`, `UNANSWERABLE_REASONS` |

Pinned to commit `096acedff9e8792b4240062b2c7c5c4b44b05f64`.

The Gazette PDFs are **vendored** into `data/reference/primary/` rather than
read out of a sibling checkout, and their SHA-256s are verified against
`MANIFEST.json` on load. A tool whose answers depend on a document must fail
loudly when the document is not the one it was built against, and that check
is itself a tool error the agent has to handle.

### The inherited limitation, stated up front

All 28 golden rows are `gazette-derived`: the slab was read out of the pinned
notification and the heading out of each authority's operative ruling. **No
human has confirmed any of them.** Every accuracy figure this project
publishes inherits that, and a before/after delta inherits it twice. The delta
is still meaningful — the same imperfect reference scores both sides — but an
absolute success rate from this suite is agreement with a document lookup, not
with a human, and will be labelled as such wherever it appears.

28 rows is also too few. §9 covers what the suite does about it.

---

## 9. The task suite

Target **40–60 scenarios**. The 28 golden rows supply the classification core;
each becomes an invoice line by attaching a declared HSN, a declared rate, a
taxable value and a date. Those four fields are **constructed, not observed**,
and the construction is deterministic from the row id so the suite is
reproducible.

The remaining scenarios exist to populate branches the golden set cannot
reach on its own:

- **out-of-scope** lines (alcoholic liquor) — terminal stop before any lookup
- **under-specified** lines — `UNANSWERABLE` with each reason code
- **pre-22-September-2025** invoice dates — outside the archive, refusal
- **1 February 2026 boundary** lines on headings 2401–2404, where the correct
  answer differs on either side of the date, and biris split from the rest of
  tobacco
- **ambiguous-heading** lines (7418, 8711, 2202) — force `check_conditions`

Constructed scenarios are marked `synthetic: true` and scored in their own
column. A success rate that mixes 28 real classification problems with 20
scenarios I wrote to exercise my own branches is not one number, and will not
be reported as one.

---

## 10. Bounds

Hard, set now, before the first chaos run.

| Bound | Value | Why |
|---|---|---|
| max iterations | 12 | Longest legitimate path is 7 tools + 2 retries + draft |
| max tool calls | 20 | |
| max calls per tool | 4 | Catches retry storms on one tool specifically |
| max tokens per run | 60 000 | |
| max wall-clock per run | 120 s | Raised to 600 s under the `slow` chaos mode, which is the point of that mode |
| max wall-clock per suite | 45 min | The stop that prevents an overnight bill |

Exceeding any bound ends the run in `budget_exhausted`, which is a failure and
is reported separately from a wrong answer.

---

## 11. Reversals

Appended, not edited over the top. See the note at the head of this document.

### 2026-09-06 — wall-clock per run raised from 120 s to 600 s

**What happened.** The first full baseline run against
`nvidia/nemotron-3-ultra-550b-a55b`, at six-way concurrency, ended
`budget_exhausted` on **15 of the first 15 derived scenarios**, every one of
them on `max_wall_clock_s`. Not one produced an answer to score.

**Why the bound was wrong.** 120 s was set in week 1 against no measured model
latency. It turns out to bound the *provider*, not the agent. A derived
scenario needs five or six model turns; the 550B endpoint returns in 2–7 s
unloaded but 20–40 s under concurrency, so the wall clock is consumed before
the agent has done anything wrong. The other three bounds — 12 iterations, 20
tool calls, 60 000 tokens — already constrain agent behaviour precisely and did
not trip once.

This is the same error the loop already had with 503s and had fixed: charging
infrastructure cost to the agent, and then recording the result as the agent
giving up. It was simply baked into a pinned number rather than into code.

**Evidence it is latency and not looping.** The same scenarios at concurrency 1
complete in 25–55 s. The two-step synthetic scenarios passed throughout, even
taking 110–172 s, because the bound is checked at the top of each iteration and
a two-step run only checks it twice — which is also why the smoke test looked
healthy and the real run did not.

**What changes.** `max_wall_clock_s` 120 → 600, and the `slow` chaos mode's
ceiling 600 → 1200 so that mode still means something. Nothing else moves.
Wall-clock stays a backstop against a hang; iterations, tool calls and tokens
remain the bounds that describe the agent.

**What this costs.** A genuinely hung run now occupies a worker for ten minutes
instead of two. The suite-level 45-minute stop is unchanged and is what
actually prevents an overnight bill.

**Recorded, not hidden:** every published run states the budget it ran under,
so a figure produced at 120 s and one produced at 600 s can never be compared
by accident.
