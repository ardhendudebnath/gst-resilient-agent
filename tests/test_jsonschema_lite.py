"""The validator's one indispensable property: it never under-checks silently."""

from __future__ import annotations

import pytest

from agent.jsonschema_lite import UnsupportedSchema, check_schema, validate


def test_unsupported_keyword_is_refused_at_registration():
    """The failure mode this guards against is a constraint nobody enforces."""
    with pytest.raises(UnsupportedSchema, match="oneOf"):
        check_schema({"type": "object", "oneOf": [{"type": "string"}]})


def test_unsupported_keyword_is_refused_inside_properties():
    with pytest.raises(UnsupportedSchema, match=r"properties\.h"):
        check_schema({"type": "object", "properties": {"h": {"format": "date"}}})


def test_unknown_type_is_refused():
    with pytest.raises(UnsupportedSchema, match="unknown type"):
        check_schema({"type": "decimal"})


def test_annotations_are_allowed():
    check_schema({"type": "string", "description": "a heading", "default": "6810"})


SCHEMA = {
    "type": "object",
    "properties": {
        "heading": {"type": "string", "pattern": r"^\d{4}$"},
        "value": {"type": "number", "minimum": 0},
        "slab": {"type": "string", "enum": ["5", "18", "40"]},
    },
    "required": ["heading"],
    "additionalProperties": False,
}


def test_valid_arguments_pass():
    assert validate({"heading": "6810", "value": 10.5, "slab": "18"}, SCHEMA) == []


def test_missing_required_is_reported():
    assert "missing required property 'heading'" in validate({}, SCHEMA)[0]


def test_unexpected_property_is_reported():
    """Defaults to closed: an invented argument means the tool was misunderstood."""
    problems = validate({"heading": "6810", "hedaing": "6810"}, SCHEMA)
    assert any("unexpected property 'hedaing'" in p for p in problems)


def test_pattern_and_enum_and_minimum():
    problems = validate({"heading": "68", "value": -1, "slab": "12"}, SCHEMA)
    joined = " ".join(problems)
    assert "does not match" in joined
    assert "below minimum" in joined
    assert "'12' not in" in joined


def test_boolean_is_not_an_integer():
    """True == 1 in Python and is almost never an integer in a schema."""
    assert validate(True, {"type": "integer"}) == ["$: expected integer, got boolean"]


def test_arrays_are_validated_elementwise():
    schema = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    assert validate(["a"], schema) == []
    assert "expected string" in validate(["a", 2], schema)[0]
    assert "fewer than minItems" in validate([], schema)[0]


def test_problems_are_returned_not_raised():
    """The agent has to read what was wrong with its call and fix it."""
    problems = validate({"heading": 6810}, SCHEMA)
    assert isinstance(problems, list) and problems
