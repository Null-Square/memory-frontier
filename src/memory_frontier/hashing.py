from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Iterable


SCIENTIFIC_HASH_SCHEME = "decimal-string-v1"
SCIENTIFIC_HASH_FLOAT_DECIMALS = 12


def _normalize_scientific_value(value: Any, *, float_decimals: int) -> Any:
    """Canonicalize a JSON-like value for tolerance-aware scientific hashing.

    Floating-point results are represented as fixed-width decimal strings. This
    intentionally ignores sub-tolerance solver noise while leaving integers,
    strings, booleans, nulls, and discrete algorithm structure exact.
    """
    if isinstance(value, dict):
        return {
            str(key): _normalize_scientific_value(item, float_decimals=float_decimals)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_scientific_value(item, float_decimals=float_decimals)
            for item in value
        ]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific hashes do not permit non-finite floats")
        # Canonicalize negative zero and values that round to zero.
        zero_threshold = 0.5 * 10.0 ** (-float_decimals)
        canonical = 0.0 if abs(value) < zero_threshold else value
        return format(canonical, f".{float_decimals}f")

    # NumPy scalar values and similar numeric wrappers expose item(). Avoid a
    # hard NumPy dependency in this small hashing utility.
    item = getattr(value, "item", None)
    if callable(item):
        unpacked = item()
        if unpacked is not value:
            return _normalize_scientific_value(
                unpacked, float_decimals=float_decimals
            )
    raise TypeError(f"unsupported value in scientific hash payload: {type(value)!r}")


def scientific_sha256(
    payload: dict[str, Any],
    *,
    exclude_keys: Iterable[str] = (),
    float_decimals: int = SCIENTIFIC_HASH_FLOAT_DECIMALS,
    scheme: str = SCIENTIFIC_HASH_SCHEME,
) -> str:
    """Tolerance-aware SHA-256 for a scientific JSON payload.

    The digest is over an explicit envelope containing the normalization scheme,
    decimal precision, and normalized payload. ``exclude_keys`` is useful for
    omitting a card's legacy raw digest from its portable scientific digest.
    """
    if float_decimals < 0:
        raise ValueError("float_decimals must be non-negative")
    excluded = set(exclude_keys)
    body = {key: value for key, value in payload.items() if key not in excluded}
    envelope = {
        "hash_scheme": scheme,
        "float_decimals": float_decimals,
        "payload": _normalize_scientific_value(
            body, float_decimals=float_decimals
        ),
    }
    canonical = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def theory_card_scientific_sha256(card: dict[str, Any]) -> str:
    return scientific_sha256(card, exclude_keys=("theory_card_sha256",))


def finite_horizon_card_scientific_sha256(card: dict[str, Any]) -> str:
    return scientific_sha256(
        card, exclude_keys=("finite_horizon_card_sha256",)
    )
