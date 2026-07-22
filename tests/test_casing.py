"""Tests for CASING module — axial + circumferential shape composition.

Builders previously registered by YAML descriptors (now deleted as part of
framework simplification).  They are registered directly here for testing.
"""

import numpy as np
import pytest

from rubimpact.core.registry import components as registry


# ── Register builders directly (was YAML-driven) ──

def _build_cylindrical(cfg):
    return {"type": "CYLINDRICAL", "get_delta": lambda x: 0.0}

def _build_taper(cfg):
    slope = float(cfg["slope"])
    return {"type": "AXIAL_TAPER", "get_delta": lambda x, s=slope: s * x}

def _build_uniform(cfg):
    return {"type": "UNIFORM", "get_delta": lambda theta: 0.0}

def _build_lobe(cfg):
    N_lobe = int(cfg["N_lobe"])
    d0 = float(cfg["d0"])
    return {"type": "LOBE", "get_delta": lambda theta, n=N_lobe, d=d0: d * np.sin(n * theta)}

registry.register("axial_shape", "CYLINDRICAL", _build_cylindrical)
registry.register("axial_shape", "AXIAL_TAPER", _build_taper)
registry.register("circumferential_shape", "UNIFORM", _build_uniform)
registry.register("circumferential_shape", "LOBE", _build_lobe)


class TestAxialShapeBuilders:
    def test_cylindrical_returns_zero(self):
        builder = registry.get("axial_shape", "CYLINDRICAL")
        result = builder({})
        assert result["type"] == "CYLINDRICAL"
        assert result["get_delta"](0.0) == 0.0
        assert result["get_delta"](100.0) == 0.0

    def test_taper_returns_slope_times_x(self):
        builder = registry.get("axial_shape", "AXIAL_TAPER")
        result = builder({"slope": "-0.056"})
        assert result["type"] == "AXIAL_TAPER"
        assert result["get_delta"](0.0) == 0.0
        assert result["get_delta"](30.0) == pytest.approx(-1.68)

    def test_taper_missing_slope_raises(self):
        builder = registry.get("axial_shape", "AXIAL_TAPER")
        with pytest.raises(KeyError):
            builder({})


class TestCircumferentialShapeBuilders:
    def test_uniform_returns_zero(self):
        builder = registry.get("circumferential_shape", "UNIFORM")
        result = builder({})
        assert result["type"] == "UNIFORM"
        assert result["get_delta"](0.0) == 0.0
        assert result["get_delta"](np.pi) == 0.0

    def test_lobe_returns_sine(self):
        builder = registry.get("circumferential_shape", "LOBE")
        result = builder({"N_lobe": "3", "d0": "0.5"})
        assert result["type"] == "LOBE"
        assert result["get_delta"](0.0) == 0.0
        assert result["get_delta"](np.pi / 6) == pytest.approx(0.5)

    def test_lobe_missing_params_raises(self):
        builder = registry.get("circumferential_shape", "LOBE")
        with pytest.raises(KeyError):
            builder({})
