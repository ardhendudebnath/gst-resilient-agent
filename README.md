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

Not yet measured. The chaos harness (`chaos/`) is not built. This table is the
deliverable of weeks 4–7 and will report success rate at 0%, 10%, 25% and 50%
failure-injection rates, before and after fixes.

| Condition | Baseline | After fixes | Δ |
|---|---|---|---|
| Clean (no chaos) | — | — | — |
| 10% failure injection | — | — | — |
| 25% failure injection | — | — | — |
| 50% failure injection | — | — | — |
| Injection attacks | — | — | — |

## The failure taxonomy

`FAILURES.md` does not exist yet. Target is 12–20 documented classes, each with
its trigger, frequency, root cause, fix and residual risk.

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
pip install -e '.[gazette,models,dev]'
cp .env.example .env    # add ANTHROPIC_API_KEY
```

Run the worked example from `docs/DESIGN.md` §1:

```bash
python -m agent --demo
```

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

- **The chaos harness, the task suite, the MCP server and the trace viewer do
  not exist.** `chaos/`, `suite/` and `mcp_gazette/` are empty packages.
- **`check_pass` and `recovery_policies` are declared and unimplemented.**
  Switching either on raises rather than silently doing nothing.
- **`check_conditions` covers five headings** (7418, 8711, 2202, 9608, 2403).
  Any other ambiguous heading returns `not_covered`, and the correct response is
  a refusal. There is no general condition-resolver and there could not be.
- **Candidate recall caps the suite at ~82%.** In 5 of 28 golden rows the gold
  heading appears in neither `propose_headings` channel, so those lines cannot
  be answered correctly however good the reasoning is.
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
