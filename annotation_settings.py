"""Small annotation preferences; video geometry stays in exported mask recipes."""
from __future__ import annotations

import copy
import math
import re

from direction import DirectionRanges
from occlusion_3d import LengthReference, UNITS


def settings_section(settings, key):
    value = settings.get(key, {})
    return value if isinstance(value, dict) else {}


def preference_choice(value, choices, default):
    return value if isinstance(value, str) and value in choices else default


def preference_number(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value)) if math.isfinite(value) else default


def reference_preferences(references):
    return [{"id": ref.id, "name": ref.name, "length": ref.length, "unit": ref.unit} for ref in references]


def restore_references(entries):
    if not isinstance(entries, list):
        return []
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key, name = entry.get("id"), entry.get("name")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key) or key in seen:
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        unit = preference_choice(entry.get("unit"), UNITS, "cm")
        length = preference_number(entry.get("length"), 35 / (UNITS[unit] * 100), 1e-8 / UNITS[unit], 100000 / UNITS[unit])
        result.append(LengthReference(name.strip(), length, unit, id=key))
        seen.add(key)
    return result


_RANGE_KEYS = ("object_margin", "near_margin", "mode", "unit", "reference_id", "top_reference_id", "bottom_reference_id")


def range_preferences(spec):
    result = {key: getattr(spec, key) for key in _RANGE_KEYS}
    result["mode_values"] = copy.deepcopy(getattr(spec, "_mode_values", {}))
    return result


def restore_ranges(data, base=None, *, three_d=False):
    result = copy.deepcopy(base) if base is not None else DirectionRanges()
    data = data if isinstance(data, dict) else {}
    result.unit = preference_choice(data.get("unit"), UNITS, result.unit)
    result.mode = preference_choice(data.get("mode"), {"pixels", "geometric", "physical"}, result.mode)
    if three_d and result.mode == "geometric":
        result.mode = "pixels"
    limit = 100000 / UNITS[result.unit] if result.mode == "physical" else 300
    for key in ("object_margin", "near_margin"):
        value = preference_number(data.get(key), getattr(result, key), -limit, limit)
        setattr(result, key, value if result.mode == "physical" else round(value))
    for key in ("reference_id", "top_reference_id", "bottom_reference_id"):
        value = data.get(key, getattr(result, key))
        if value is None or isinstance(value, str):
            setattr(result, key, value)
    profiles = data.get("mode_values", {})
    result._mode_values = {}
    if isinstance(profiles, dict):
        for mode, values in profiles.items():
            if mode in {"pixels", "geometric", "physical"} and isinstance(values, (list, tuple)) and len(values) == 2:
                bound = 100000 / UNITS[result.unit] if mode == "physical" else 300
                result._mode_values[mode] = tuple(preference_number(v, default, -bound, bound) for v, default in zip(values, (0, 2 if mode == "physical" else 20)))
    return result
