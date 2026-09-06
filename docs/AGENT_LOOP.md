# The agent loop, line by line

`agent/loop.py::run_task` (line 424). Roughly 130 lines of actual control flow,
no framework - the repository has no LangChain, LangGraph, LlamaIndex, AutoGen
or CrewAI dependency, and `grep` will confirm it.

That is a deliberate choice and it is not about purity. Every abstraction a
framework hides here is somewhere this project needs to **inject a failure and
measure the recovery**: how a tool result re-enters the context, what happens
when the model emits something unparseable, which bound trips first, whether a
retry is charged to the agent or to the provider. You cannot chaos-test a call
you cannot see.

---

## The components

```
run_task                                     agent/loop.py:424
├── LLMClient .......... Model protocol      agent/llm.py:105 -> OpenAICompatModel
├── ToolRegistry ....... Registry.invoke     agent/registry.py:168
├── StateManager ....... messages + RunState agent/loop.py:453, :298
├── RetryPolicy ........ is_transient +
│                        backoff_delay       agent/llm.py:143, :157
├── Validation ......... jsonschema_lite     agent/jsonschema_lite.py (arguments in)
│                        opinion.validate    agent/opinion.py         (object out)
├── RecoveryPolicy ..... RecoveryPolicy      agent/recovery.py
└── ExecutionBudget .... Budget + Ledger     agent/budget.py
```

`StateManager` is two things on purpose. `messages` is what the model sees;
`RunState` is what the *loop* knows - which tools succeeded, and whether a slab
has been established by a tool rather than asserted in prose. The prerequisite
checks read the second and never the first, because an agent that can talk its
way past a precondition has no precondition.

---

## Setup, before the loop

```python
policy   = policy or Policy.baseline()      # which defences are on
recovery = RecoveryPolicy() if policy.recovery_policies else None
ledger   = Ledger(budget=budget or Budget())
cache    = IdempotencyCache()               # per run, never shared
state    = RunState()
messages = [Message("user", render_line_item(line))]
```

Four of these are **per run**, and that is load-bearing. A cache shared between
runs would carry a result injected under 50% chaos into a clean run and corrupt
the baseline. A `RecoveryPolicy` shared between runs would let one task spend
another's retry budget.

The invoice line goes in the **first user turn**, never the system prompt. That
is the one injection defence that is structural rather than switchable: no
document text can occupy a system-prompt position, and a test asserts it.

---

## The loop

### 1. Check the bounds *before* spending - `:508`

```python
if broke := ledger.exceeded():
    reason = broke
    return finish()
step = ledger.iterations + 1
```

`exceeded()` returns the *name* of the bound that broke, or None. Checked at the
top, so a bound is a refusal rather than a post-mortem. Which one tripped is
recorded, because "ran out of steps" and "ran out of money" need different
fixes:

| bound | value | what it actually bounds |
|---|---:|---|
| `max_iterations` | 12 | the agent's reasoning |
| `max_tool_calls` | 20 | the agent's reasoning |
| `max_calls_per_tool` | 4 | a retry storm on one tool (enforced in the registry) |
| `max_tokens` | 150 000 | spend - see DESIGN §11 |
| `max_wall_clock_s` | 600 | a hang - see DESIGN §11 |

Note `step` is `iterations + 1`, not `iterations`. The counter is incremented
later, at `:562`, and only once a reasoning step has actually happened. Why is
the subject of step 3.

### 2. Ask the model - `:513`-`:528`

```python
tracer.emit("llm_call", step=step, messages=len(messages), ...)
completion = model.complete(system_prompt, messages)
ledger.note_tokens(completion.tokens_in, completion.tokens_out)
tracer.emit("llm_result", step=step, ok=..., text=..., tokens_in=..., ...)
```

Both sides are traced. Tokens are counted whether or not the call succeeded,
because a failed call still bills.

`model` is the `Model` protocol, so this line is identical for
`OpenAICompatModel` (NVIDIA, NIM, OpenRouter, OpenAI), `AnthropicModel` and
`ScriptedModel`. That is why the test suite can exercise the whole loop with no
key and no network.

### 3. A provider failure is not a reasoning step - `:530`-`:559`

```python
if not completion.ok:
    model_errors += 1
    if not is_transient(completion.error):
        reason = f"model_error: {completion.error}"; return finish()
    if model_errors > MAX_MODEL_RETRIES:
        reason = f"model_unavailable: {completion.error}"; return finish()
    time.sleep(_backoff(model_errors))
    continue                       # <- note: no note_iteration()
```

This block exists because of a measured bug. The 550B endpoint returns
`503 Service temporarily overloaded` often - one eight-step run met three - and
an earlier version charged each to `iterations`. A longer line would have ended
`budget_exhausted`, **recording an infrastructure failure as the agent giving
up**. Worse, once chaos testing starts, an organic 503 counted that way is
indistinguishable from an injected one.

So transient errors retry with jittered backoff and are never charged to
`iterations`; non-transient ones (401, malformed request) stop immediately
rather than retrying to reach the same failure more slowly; and the count is
reported as `model_retries`, so a suite quietly absorbing 503s is visible rather
than silently flattering.

### 4. Only now is it a step - `:562`

```python
ledger.note_iteration()
```

One line, and it is the fix from step 3. `iterations` counts *reasoning*, not
attempts.

### 5. Parse the action - `:564`-`:586`

```python
action, problem = parse_action(completion.text)
messages.append(Message("assistant",
    completion.text if problem is not None else compact_action(action)))
```

The protocol is one JSON object: `{"thought", "tool", "arguments"}`. Hand-parsed
rather than using provider-native tool calling - native tool use is more
reliable, and that is exactly the problem: it removes the output parser, and the
parser is an attack surface the brief names ("content that breaks your output
parser"). A formatting attack needs something to break.

`parse_action` tries, in order: the raw string, a fenced block, then outermost
balanced braces (scanned, not regexed - JSON nests and a regex cannot count). It
never raises; it returns a *specific* problem string, because "invalid JSON" is
not something a model can act on.

**`compact_action` is why the assistant turn is not stored verbatim.** Measured:
the model spent 1,000-1,900 tokens on replies whose payload needs about 150, and
every assistant message is resent on every later turn - a 1,894-token reply was
followed by an input that grew by 2,073. The history keeps the parsed action;
the trace keeps the full text, because diagnosis needs it. An *unparseable*
reply is resent verbatim, because the correction that follows refers to it.

Parse failures are bounded by `policy.max_parse_retries` (2) and each still
costs an iteration, so a model that only ever emits prose terminates.

### 6. The stage allowlist - `:590`-`:603`

```python
if policy.stage_allowlist and (blocked := check_prerequisites(action.tool, state)):
    tracer.emit("defence", ...)
    messages.append(Message("user", f"TOOL CALL REFUSED: {blocked}. ..."))
    continue
```

Off in the baseline. Prerequisites rather than strict stage ordering, because
skipping is legitimate - a heading that resolves cleanly never needs
`check_conditions`, and refusing that call would break the happy path in the
name of defending it.

What it actually blocks is the tool-misuse injection: `draft_opinion` requires a
slab that a **tool** established (`RunState.has_resolved_slab`), not one asserted
in a message. A document saying "use correct_slab = 0" cannot satisfy that.

### 7. Execute - `:605`-`:606`

```python
call   = build_call(action.tool, action.arguments, step=step)
result = dispatcher.invoke(call, ledger=ledger, tracer=tracer, cache=cache)
```

Two lines, and everything the contract promises happens inside `invoke`:
arguments validated against the tool's JSON Schema; the per-tool cap enforced;
repeats served from the idempotency cache; exceptions converted to
`error="internal"`; the call and result traced with timing; and the result
stamped with `tool_call_id` and `latency_ms`.

`dispatcher` is typed as a `Protocol`, not as `Registry`. That is what lets
`ChaosDispatcher` wrap it with the same signature - **the loop cannot tell which
one it is holding**, and an agent that could detect it was being tested would
not be being tested.

`ToolCall` carries two identifiers and conflating them is a real bug: `call_id`
is unique per invocation, `key` is the hash of `(name, arguments)` and is shared
by every repeat. Deduplication keys on the second; the trace keys on the first.

### 8. Recover, by failure class - `:608`-`:650`

```python
if recovery is not None and not result.ok:
    while True:
        decision = recovery.decide(call.key, result)
        tracer.emit("policy", ...)
        if decision.action != RETRY:
            advice = decision.guidance
            if decision.action == ABORT: ... return finish()
            break
        time.sleep(backoff_delay(...))
        call   = build_call(action.tool, action.arguments, step=step)
        result = dispatcher.invoke(call, ledger=ledger, tracer=tracer, cache=cache)
        if result.ok: break
```

Off in the baseline. `timeout` / `rate_limited` / `unavailable` are re-issued
**here**, without a model turn - routing a timeout through the model spends an
iteration and several thousand tokens of resent history to reach the decision
the error code already implied. `bad_argument`, `malformed` and `internal` get
class-specific guidance appended to the result. `source_mismatch` aborts: no
number of retries conjures the right Gazette.

The re-issue goes through `dispatcher.invoke` with the same `ledger`, so
`max_calls_per_tool` still applies. **A recovery policy that could bypass the cap
would be the retry storm it exists to prevent.**

### 9. Feed the result back - `:653`-`:659`

```python
visible  = result if result.chaos is None else result.with_chaos(None)
rendered = render_tool_result(action.tool, visible,
                              quarantine_evidence=policy.quarantine_evidence)
if advice: rendered += f"\n\nRECOVERY: {advice}"
messages.append(Message("user", rendered))
```

The chaos label is stripped **here**, at the one point a result crosses into the
message history. `render_tool_result` asserts if a labelled result reaches it,
so a wiring mistake is loud rather than silent.

`render_tool_result` keeps `data` (trusted, computed here) and `evidence`
(verbatim document text) in separate blocks. In the baseline the evidence is
rendered inline and undelimited - that is the vulnerability, rendered honestly,
and a test asserts the baseline **stays** vulnerable so a defence cannot leak
into the "before".

### 10. Stop - `:661`-`:665`

```python
if action.tool == "draft_opinion" and result.ok:
    opinion  = result.data.get("opinion")
    terminal = str(result.data.get("terminal") or "opinion")
    justification_source = result.data.get("justification_source")
    return finish()
```

**Every path ends through `draft_opinion`, including the refusals.** So the
output schema gates every terminal and there is no second, unvalidated way to
finish. A run that drafts an invalid opinion - 28% on a March 2026 invoice -
gets `malformed` back and re-drafts.

Four terminals: `opinion`, `out_of_scope`, `unanswerable`, and
`budget_exhausted`. The last is always a failure and is reported *apart* from a
wrong answer, because running out of steps and answering incorrectly are
different faults with different fixes.

---

## What every run produces

| field | where |
|---|---|
| `run_id` | `Tracer.run_id`, on the run and every trace event |
| `tool_call_id` | `ToolCall.call_id`, stamped onto the result by `Registry._finish` |
| timestamps | `run_start.started_at` (UTC), `t_ms` on every event |
| tool arguments | `tool_call` event |
| tool result | `tool_result` event, full `ToolResult.to_json()` |
| latency | `latency_ms` on the result; `duration_ms` on the event |
| error | closed vocabulary, `agent/contract.py::ERROR_CODES` |
| final status | `run_end.terminal` + `reason` |

The trace is **JSONL, flushed per event**. Not a single JSON document written at
the end, and the reason is specific to this project: the whole point of week 4
is to make runs die in the middle. A half-written JSON array is unparseable and
takes its evidence with it; a half-written JSONL file is readable up to the last
complete line - which is exactly the line before the thing you are diagnosing.

---

## What this loop does not do

- **No `check_pass`.** Declared in `agent/policy.py`, unbuilt, and it *raises* if
  switched on rather than quietly doing nothing. A flag reporting a defence as
  active while nothing happens would flatter the after-fix numbers.
- **No concurrency inside a run.** One run is one sequence of steps; `Tracer` is
  documented as not thread-safe and there is one per task. The suite
  parallelises across tasks instead.
- **No streaming.** The action is one object; there is nothing to stream.
- **No context compaction beyond `compact_action`.** When a run exhausts
  `max_tokens` it stops rather than silently dropping history, because a run
  that quietly forgot its own evidence is worse than one that admits it ran out.
