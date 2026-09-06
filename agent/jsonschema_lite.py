"""A JSON Schema subset, validated strictly, with no dependencies.

The brief suggests reaching for a library for schema validation, and for most
projects that is right. Here the core is deliberately stdlib-only — `make test`
has to pass on a fresh clone, and a reliability harness whose own dependencies
can fail is measuring the wrong thing. So this exists instead, and it is
narrow on purpose.

**The property that makes a 120-line validator acceptable: it refuses what it
does not understand.** Every keyword it meets is either implemented or raises
`UnsupportedSchema` at registration time, when a human is watching. There is no
path on which an unrecognised constraint is quietly skipped and a tool receives
arguments nobody checked. A validator that silently under-checks is worse than
no validator, because it is believed.

Supported: `type` (object/string/integer/number/boolean/array/null),
`properties`, `required`, `additionalProperties`, `enum`, `pattern`,
`minimum`/`maximum`, `minLength`/`maxLength`, `items`, `minItems`/`maxItems`,
`description`/`title`/`default`/`examples` (annotations, ignored).

Not supported, and loudly: `oneOf`, `anyOf`, `allOf`, `not`, `$ref`, `format`,
`patternProperties`, `dependentRequired`, `const`. If a tool needs one, add it
here with a test rather than working around it.
"""

from __future__ import annotations

import re
from typing import Any

#: Keywords that carry documentation, not constraints. Skipped knowingly.
_ANNOTATIONS = frozenset({"description", "title", "default", "examples", "$comment"})

_SUPPORTED = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "pattern",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "items",
        "minItems",
        "maxItems",
    }
) | _ANNOTATIONS

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "null": (type(None),),
}


class UnsupportedSchema(RuntimeError):
    """A schema uses a keyword this validator cannot check.

    Raised at registration, never at call time: a tool whose arguments cannot
    be fully checked must not reach production, and finding out mid-run is too
    late to do anything about it.
    """


def check_schema(schema: dict[str, Any], *, where: str = "schema") -> None:
    """Assert that every keyword in `schema` is one we actually enforce."""
    if not isinstance(schema, dict):
        raise UnsupportedSchema(f"{where}: expected an object, got {type(schema).__name__}")
    for key in schema:
        if key not in _SUPPORTED:
            raise UnsupportedSchema(
                f"{where}: keyword {key!r} is not implemented by jsonschema_lite. "
                "Implement it with a test, or restate the constraint using a "
                "supported keyword — do not leave it unchecked."
            )
    if (t := schema.get("type")) is not None and t not in _TYPES:
        raise UnsupportedSchema(f"{where}: unknown type {t!r}; known: {sorted(_TYPES)}")
    for name, sub in (schema.get("properties") or {}).items():
        check_schema(sub, where=f"{where}.properties.{name}")
    if (items := schema.get("items")) is not None:
        check_schema(items, where=f"{where}.items")


def validate(value: Any, schema: dict[str, Any], *, path: str = "") -> list[str]:
    """Return human-readable problems with `value`. Empty means valid.

    Returns rather than raises: a tool given bad arguments reports
    `error="bad_argument"` with these strings attached, because the agent has
    to be able to read what was wrong with its call and fix it. A traceback
    tells it nothing it can act on.
    """
    errs: list[str] = []
    at = path or "$"

    # -- type -------------------------------------------------------------
    t = schema.get("type")
    if t is not None:
        expected = _TYPES[t]
        # bool is an int in Python and almost never is one in a schema.
        if t in ("integer", "number") and isinstance(value, bool):
            return [f"{at}: expected {t}, got boolean"]
        if not isinstance(value, expected):
            got = type(value).__name__
            return [f"{at}: expected {t}, got {got}"]

    # -- enum -------------------------------------------------------------
    if (allowed := schema.get("enum")) is not None and value not in allowed:
        errs.append(f"{at}: {value!r} not in {allowed}")

    # -- strings ----------------------------------------------------------
    if isinstance(value, str):
        if (p := schema.get("pattern")) is not None and not re.search(p, value):
            errs.append(f"{at}: {value!r} does not match /{p}/")
        if (n := schema.get("minLength")) is not None and len(value) < n:
            errs.append(f"{at}: shorter than minLength {n}")
        if (n := schema.get("maxLength")) is not None and len(value) > n:
            errs.append(f"{at}: longer than maxLength {n}")

    # -- numbers ----------------------------------------------------------
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if (n := schema.get("minimum")) is not None and value < n:
            errs.append(f"{at}: {value} below minimum {n}")
        if (n := schema.get("maximum")) is not None and value > n:
            errs.append(f"{at}: {value} above maximum {n}")

    # -- arrays -----------------------------------------------------------
    if isinstance(value, (list, tuple)):
        if (n := schema.get("minItems")) is not None and len(value) < n:
            errs.append(f"{at}: fewer than minItems {n}")
        if (n := schema.get("maxItems")) is not None and len(value) > n:
            errs.append(f"{at}: more than maxItems {n}")
        if (items := schema.get("items")) is not None:
            for i, item in enumerate(value):
                errs += validate(item, items, path=f"{at}[{i}]")

    # -- objects ----------------------------------------------------------
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errs.append(f"{at}: missing required property {name!r}")
        # Defaults to closed. An agent that invents an argument has
        # misunderstood the tool, and silently dropping it hides that in
        # exactly the runs where it matters.
        if schema.get("additionalProperties", False) is False:
            for name in value:
                if name not in props:
                    errs.append(f"{at}: unexpected property {name!r}")
        for name, sub in props.items():
            if name in value:
                errs += validate(value[name], sub, path=f"{at}.{name}")

    return errs
