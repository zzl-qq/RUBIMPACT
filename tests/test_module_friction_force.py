"""Tests for protocolized friction force module."""
import numpy as np
from rubimpact.modules.friction_force import FrictionForceModule
from rubimpact.core.registry import components


def test_friction_force_registered():
    assert components.resolve_module("friction_force", "COULOMB") is not None
    assert components.resolve_module("friction_force", "STRIBECK") is not None


def test_coulomb_protocol():
    mod = FrictionForceModule(None, {})
    mod.configure({"TYPE": "COULOMB", "mu": "0.3"})
    proto = mod.get_pipeline_protocol()
    assert proto.stages[0].name == "compute_friction_force"
    assert proto.stages[0].kernel_ref == "friction/COULOMB"
    assert proto.stages[0].depends_on == ["compute_normal_force"]
    assert proto.params["fric_params"][0] == 0.3


def test_stribeck_protocol():
    mod = FrictionForceModule(None, {})
    mod.configure({"TYPE": "STRIBECK", "mu_s": "0.4", "mu_d": "0.3", "v_s": "2.0"})
    proto = mod.get_pipeline_protocol()
    assert proto.stages[0].kernel_ref == "friction/STRIBECK"
    assert len(proto.params["fric_params"]) == 3
