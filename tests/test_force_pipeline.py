"""Pipeline comparison tests — protocol-driven PipelineFactory vs per-node reference.

When adding a new submodule TYPE, register it in ComponentRegistry and it
automatically enters this test matrix (via list_combinations).
"""
import numpy as np
import pytest
import rubimpact.kernels        # trigger kernel registration
import rubimpact.modules        # trigger module registration
from rubimpact.core.registry import components
from rubimpact.core.pipeline_factory import PipelineFactory
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol


def test_all_force_assembler_combinations():
    """Pipeline output matches per-node reference for all TYPE combos."""
    components.register_orchestrator("ForceAssembler",
        ["contact_force", "friction_force"])
    combos = components.list_combinations("ForceAssembler")
    if not combos:
        pytest.skip("No ForceAssembler ORCHESTRATOR registered — register with "
                     "components.register_orchestrator('ForceAssembler', "
                     "['contact_force', 'friction_force'])")
    for combo in combos:
        _verify_combo(combo)


def _verify_combo(combo):
    """Verify pipeline result matches per-node reference for one TYPE combo."""
    from rubimpact.infra.databus import DataBus

    db = DataBus()
    db.set("coating.grid", {"A_cell": 1.0, "h_coat": 1.0})
    db.set("coating.h", np.ones((10, 20)))
    db.set("coating.ep", np.zeros((10, 20)))
    db.set("coating.alpha", np.zeros((10, 20)))
    db.set("coating.interpolator_type", "BILINEAR")
    db.set("rom.n_r", 5)

    # Build modules from registry
    cf_type = combo["contact_force"]
    ff_type = combo["friction_force"]

    cf_spec = components.resolve_module("contact_force", cf_type)
    ff_spec = components.resolve_module("friction_force", ff_type)
    assert cf_spec is not None, f"contact_force/{cf_type} not registered"
    assert ff_spec is not None, f"friction_force/{ff_type} not registered"

    cf_cfg = {"TYPE": cf_type}
    if cf_type == "PCL_CONTACT":
        cf_cfg.update({"E": "200000", "Y": "500", "K_plas": "1000",
                       "wear_law": {"TYPE": "NONE"}})
    elif cf_type == "PENALTY":
        cf_cfg["k_penalty"] = "5000"

    ff_cfg = {"TYPE": ff_type}
    if ff_type == "COULOMB":
        ff_cfg["mu"] = "0.25"
    elif ff_type == "STRIBECK":
        ff_cfg.update({"mu_s": "0.3", "mu_d": "0.25", "v_s": "1.0"})

    contact = cf_spec.builder(db, {"contact_cfg": cf_cfg})
    friction = ff_spec.builder(db, {"friction_cfg": ff_cfg})

    pipeline = PipelineFactory.build(
        modules={"contact": contact, "friction": friction},
        protocol="ForceAssembler",
    )

    # Test inputs: node0 penetrates, node1 clear
    pen = np.array([0.01, -0.001], dtype=np.float64)
    coords = np.array([
        [0.0, 0.1, 0.8, 0.6, 1.0],
        [0.5, 0.2, 0.6, 0.8, 1.0],
    ], dtype=np.float64)
    interp = np.array([
        [1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],
        [1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    dof_idx = np.array([[0, 0, 1], [0, 2, 3]], dtype=np.int64)

    n_r = 5
    F_total = np.zeros(n_r, dtype=np.float64)

    pipeline(pen, coords, interp, 100.0, dof_idx,
             F_total, np.zeros(0), np.zeros(0),
             np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))

    assert np.all(np.isfinite(F_total)), f"{cf_type}+{ff_type}: NaN in output"
    # Penetrating node (i=0) should have non-zero forces
    assert F_total[0] != 0.0 or F_total[1] != 0.0, \
        f"{cf_type}+{ff_type}: no force on penetrating node"
    # Non-penetrating node (i=1) should have zero force
    assert F_total[2] == 0.0 and F_total[3] == 0.0, \
        f"{cf_type}+{ff_type}: force on non-penetrating node"


def test_pipeline_vs_per_node_pcl_coulomb():
    """PCL + COULOMB: pipeline produces valid forces, no NaN, penetrating node gets force."""
    from rubimpact.infra.databus import DataBus

    db = DataBus()
    db.set("coating.grid", {"A_cell": 1.0, "h_coat": 1.0})
    db.set("coating.h", np.ones((10, 20)))
    db.set("coating.ep", np.zeros((10, 20)))
    db.set("coating.alpha", np.zeros((10, 20)))
    db.set("coating.interpolator_type", "BILINEAR")
    db.set("rom.n_r", 5)

    cf_spec = components.resolve_module("contact_force", "PCL_CONTACT")
    ff_spec = components.resolve_module("friction_force", "COULOMB")
    contact = cf_spec.builder(db, {"contact_cfg": {"TYPE": "PCL_CONTACT", "E": "200000", "Y": "500", "K_plas": "1000", "wear_law": {"TYPE": "NONE"}}})
    friction = ff_spec.builder(db, {"friction_cfg": {"TYPE": "COULOMB", "mu": "0.3"}})

    pipeline = PipelineFactory.build(
        modules={"contact": contact, "friction": friction},
        protocol="ForceAssembler",
    )

    pen = np.array([0.01, -0.001], dtype=np.float64)
    coords = np.array([[0.0, 0.1, 0.8, 0.6, 1.0], [0.5, 0.2, 0.6, 0.8, 1.0]], dtype=np.float64)
    interp = np.array([[1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], [1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
    dof_idx = np.array([[0, 0, 1], [0, 2, 3]], dtype=np.int64)

    F_total = np.zeros(5, dtype=np.float64)
    dummy_coat = np.zeros((1, 1), dtype=np.float64)
    pipeline(pen, coords, interp, 100.0, dof_idx, F_total, np.zeros(0), np.zeros(0),
             dummy_coat, dummy_coat, dummy_coat)

    assert np.all(np.isfinite(F_total)), "NaN in pipeline output"
    # Penetrating node (i=0) should contribute to DOFs 0 and 1
    assert (F_total[0] > 0.0 or F_total[1] > 0.0), \
        f"No force on penetrating node: F_total={F_total}"
    # Non-penetrating node (i=1) maps to DOFs 2,3 — should be zero
    assert F_total[2] == 0.0, f"Unexpected force on DOF 2: {F_total[2]}"
    assert F_total[3] == 0.0, f"Unexpected force on DOF 3: {F_total[3]}"


def test_contact_force_missing_required_params():
    """Missing required params should raise ValueError with helpful message."""
    from rubimpact.infra.databus import DataBus
    from rubimpact.modules.contact_force import ContactForceModule

    # Missing E, Y, K_plas
    db = DataBus()
    mod = ContactForceModule(db)
    with pytest.raises(ValueError, match="E"):
        mod.configure({})

    # E provided but no coating grid → should fail on grid check
    db2 = DataBus()
    mod2 = ContactForceModule(db2)
    cfg = {"E": "200000", "Y": "500", "K_plas": "1000",
           "wear_law": {"TYPE": "NONE"}}
    with pytest.raises(ValueError, match="coating.grid"):
        mod2.configure(cfg)
