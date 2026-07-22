"""Parameter extraction utilities — no silent defaults.

Two-level lookup: DataBus (via db_key) → INP cfg → ValueError.
Designed to replace all ``.get(key, default)`` patterns with mandatory
parameter extraction that fails loudly when a value is missing.
"""
from __future__ import annotations
from typing import Any
import numpy as np


def required_scalar(db, cfg: dict, key: str, *,
                    db_key: str | None = None,
                    source: str = "") -> float:
    """Extract a required scalar parameter.

    Lookup order:
        1. DataBus[db_key][key]  (if db_key provided)
        2. cfg[key]              (direct INP config)

    Args:
        db: DataBus instance.
        cfg: INP config dict for this module (or any dict).
        key: Parameter name (e.g. ``"E"``, ``"Y"``, ``"k_penalty"``).
        db_key: Optional DataBus key for nested dict lookup
                (e.g. ``"const_params"`` for params stored by
                ``ConstitutiveModule``).
        source: Human-readable hint for the error message
                (e.g. ``"*CONSTITUTIVE keyword in INP file"``).

    Returns:
        float

    Raises:
        ValueError: if the parameter is not found in either location.
    """
    # 1. Try DataBus nested lookup
    if db_key:
        container = db.get(db_key, {}) or {}
        val = container.get(key)
        if val is not None:
            return float(val)

    # 2. Try direct cfg
    val = cfg.get(key)
    if val is not None:
        return float(val)

    # 3. Error
    hint = f" Provide it via {source}." if source else ""
    raise ValueError(
        f"Required parameter '{key}' not found in DataBus "
        f"or INP config.{hint}"
    )


def required_array(db, key: str, *,
                   source: str = "") -> Any:
    """Extract a required array from DataBus.

    No cfg fallback — arrays (coating grids, etc.) are always stored
    on DataBus by upstream modules.

    Args:
        db: DataBus instance.
        key: DataBus key (e.g. ``"coating.h"``, ``"coating.ep"``).
        source: Human-readable hint
                (e.g. ``"*COATING module"``).

    Returns:
        The array as-is (no copy).

    Raises:
        ValueError: if the key is not found or the value is None.
    """
    val = db.get(key)
    if val is None:
        hint = f" Set by {source}." if source else ""
        raise ValueError(
            f"Required array '{key}' not found on DataBus.{hint}"
        )
    return val
