"""Tests for protocolized contact force modules."""
import numpy as np
from rubimpact.infra.databus import DataBus
from rubimpact.modules.contact_force import ContactForceModule, PenaltyContactForceModule
from rubimpact.core.registry import components


def test_contact_force_registered():
    """PCL_CONTACT 已注册。"""
    spec = components.resolve_module("contact_force", "PCL_CONTACT")
    assert spec is not None
    assert spec.protocol == "NormalForceProtocol"


def test_penalty_registered():
    """PENALTY 已注册。"""
    spec = components.resolve_module("contact_force", "PENALTY")
    assert spec is not None
    assert spec.protocol == "NormalForceProtocol"


def test_pcl_protocol_has_normal_force_stage():
    """PCL 协议声明了 compute_normal_force 阶段。"""
    db = DataBus()
    db.set("coating.grid", {"A_cell": 0.5, "h_coat": 1.0})
    db.set("coating.h", np.ones((10, 20)))
    db.set("coating.ep", np.zeros((10, 20)))
    db.set("coating.alpha", np.zeros((10, 20)))
    db.set("coating.interpolator_type", "BILINEAR")

    mod = ContactForceModule(db, {})
    mod.configure({"E": 200e3, "Y": 500.0, "K_plas": 1000.0,
                   "wear_law": {"TYPE": "NONE"}})

    proto = mod.get_pipeline_protocol()
    assert proto is not None
    stage_names = [s.name for s in proto.stages]
    assert "compute_normal_force" in stage_names
    assert "pcl_params" in proto.params


def test_pcl_wear_disabled_by_default():
    """默认情况下 wear 不启用。"""
    db = DataBus()
    db.set("coating.grid", {"A_cell": 0.5, "h_coat": 1.0})
    db.set("coating.h", np.ones((10, 20)))
    db.set("coating.ep", np.zeros((10, 20)))
    db.set("coating.alpha", np.zeros((10, 20)))
    db.set("coating.interpolator_type", "BILINEAR")

    mod = ContactForceModule(db, {})
    mod.configure({"E": 200e3, "Y": 500.0, "K_plas": 1000.0,
                   "wear_law": {"TYPE": "NONE"}})

    proto = mod.get_pipeline_protocol()
    stage_names = [s.name for s in proto.stages]
    assert "apply_wear" not in stage_names  # wear disabled


def test_penalty_protocol_no_kernel_ref():
    """PENALTY 协议声明 kernel_ref=None（内联逻辑）。"""
    mod = PenaltyContactForceModule(None, {})
    mod.configure({"k_penalty": 1000.0})

    proto = mod.get_pipeline_protocol()
    assert proto.stages[0].kernel_ref is None
    assert proto.params["k_penalty"] == 1000.0
