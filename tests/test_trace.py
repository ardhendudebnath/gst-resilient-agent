"""The trace is the primary artefact, so its failure modes are tested first."""

from __future__ import annotations

import json

import pytest

from agent.trace import TRACE_SCHEMA, TraceError, Tracer, read_trace


def test_unknown_event_is_refused():
    """The viewer switches on the event name; an unknown one would be dropped."""
    with pytest.raises(TraceError, match="unknown trace event"):
        Tracer(memory_only=True).emit("something_new", x=1)


def test_events_are_sequenced_and_timed():
    t = Tracer(memory_only=True)
    t.emit("note", text="a")
    t.emit("note", text="b")
    assert [e["seq"] for e in t.events] == [1, 2]
    assert all("t_ms" in e for e in t.events)


def test_run_start_stamps_the_schema_version():
    t = Tracer(memory_only=True)
    t.run_start(task_id="gst-0002", chaos="none")
    assert t.events[0]["schema"] == TRACE_SCHEMA
    assert t.events[0]["task_id"] == "gst-0002"


def test_chaos_is_its_own_event(tmp_path):
    """An injected failure that looked organic would make the taxonomy fiction."""
    t = Tracer(memory_only=True)
    t.chaos(mode="timeout", target="lookup_schedule", call_id="abc")
    assert t.events[0] == {
        **t.events[0],
        "event": "chaos",
        "mode": "timeout",
        "target": "lookup_schedule",
    }


def test_written_trace_round_trips(tmp_path):
    path = tmp_path / "run.jsonl"
    with Tracer(run_id="r1", path=path) as t:
        t.run_start(task_id="gst-0002")
        t.emit("note", text="hello")
        t.run_end(terminal="opinion", passed=True)
    events = list(read_trace(path))
    assert [e["event"] for e in events] == ["run_start", "note", "run_end"]
    assert events[-1]["terminal"] == "opinion"


def test_a_truncated_final_line_is_tolerated(tmp_path):
    """A run killed mid-hang leaves a partial last line. Refusing to read the
    file because of it would discard every event before the interesting one."""
    path = tmp_path / "run.jsonl"
    with Tracer(run_id="r2", path=path) as t:
        t.run_start(task_id="gst-0002")
        t.emit("note", text="the last thing that happened")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 3, "event": "tool_ca')
    events = list(read_trace(path))
    assert len(events) == 2
    assert events[-1]["text"] == "the last thing that happened"


def test_corruption_before_the_end_is_surfaced(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"seq": 1, "event": "note"}\nNOT JSON\n{"seq": 3}\n', encoding="utf-8")
    with pytest.raises(TraceError, match="not the last line"):
        list(read_trace(path))


def test_truncation_leaves_proof_of_what_was_removed():
    t = Tracer(memory_only=True, max_text_chars=10)
    t.emit("note", text="x" * 100)
    marker = t.events[0]["text"]
    assert marker["__truncated__"] is True
    assert marker["chars"] == 100
    assert len(marker["sha256_16"]) == 16


def test_truncation_is_off_by_default():
    """A trace that dropped its evidence is worth less than a large file."""
    t = Tracer(memory_only=True)
    t.emit("note", text="x" * 5000)
    assert t.events[0]["text"] == "x" * 5000


def test_each_event_is_flushed_as_it_is_written(tmp_path):
    """A buffered trace loses exactly the tail that explains a hang."""
    path = tmp_path / "run.jsonl"
    t = Tracer(run_id="r3", path=path)
    t.emit("note", text="written")
    # Read without closing the tracer.
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["text"] == "written"
    t.close()
