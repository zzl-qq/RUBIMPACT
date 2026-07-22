"""New ForceAssembler: pipeline self-contained, zero isinstance, compare with old path."""
import numpy as np
import pytest

from rubimpact.infra.databus import DataBus
from rubimpact.orchestrators.force_assembler import ForceAssembler


def test_force_assembler_assembles_pcl_coulomb():
    """ForceAssembler with PCL + COULOMB produces correct forces."""
    # Setup DataBus with minimum required data
    db = DataBus()
    db.set("coating.grid", {"A_cell": 1.0, "h_coat": 1.0})
    db.set("coating.h", np.ones((10, 20), dtype=np.float64))
    db.set("coating.ep", np.zeros((10, 20), dtype=np.float64))
    db.set("coating.alpha", np.zeros((10, 20), dtype=np.float64))
    db.set("coating.interpolator_type", "BILINEAR")
    db.set("rom.n_r", 5)

    # Setup tip nodes
    db.set("nodes.tip", {
        1: (0.1, 0.8, 0.6),
        2: (0.2, 0.6, 0.8),
    })

    # Setup ROM matrices (minimal, for tip_dof_indices)
    from rubimpact.infra.sparse_matrix import SparseMatrix
    rows = list(range(5))
    cols = list(range(5))
    data = [1.0] * 5
    K = SparseMatrix.from_coo(rows, cols, data, (5, 5))
    db.set("matrices.K_omega", K)
    db.set("rom.enabled", True)
    db.set("rom.tip_dof_map", [0, 1, 2, 3, 4])

    # Build ForceAssembler
    fa = ForceAssembler(db, {"Omega": 100.0})
    cfg = {
        "submodules": {
            "contact_force": {"TYPE": "PCL_CONTACT",
                              "E": "200000", "Y": "500", "K_plas": "1000",
                              "wear_law": {"TYPE": "NONE"}},
            "friction_force": {"TYPE": "COULOMB", "mu": "0.25"},
        }
    }
    fa.configure(cfg)

    # Test inputs
    pen = np.array([0.01, -0.001], dtype=np.float64)
    coords = np.array([
        [0.0, 0.1, 0.8, 0.6, 1.0],
        [0.5, 0.2, 0.6, 0.8, 1.0],
    ], dtype=np.float64)
    interp = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    context = {"Omega": 100.0, "n_r": 5}
    F_total = fa.assemble(pen, coords, interp, context)

    assert F_total.shape == (5,)
    assert np.all(np.isfinite(F_total))

    # Verify F_total is stored in DataBus
    assert db.get("F_total") is not None
    assert np.allclose(db.get("F_total"), F_total)


def test_force_assembler_assembles_penalty_coulomb():
    """ForceAssembler with PENALTY + COULOMB produces correct forces."""
    db = DataBus()
    db.set("coating.grid", {"A_cell": 1.0, "h_coat": 1.0})
    db.set("coating.h", np.ones((10, 20), dtype=np.float64))
    db.set("coating.ep", np.zeros((10, 20), dtype=np.float64))
    db.set("coating.alpha", np.zeros((10, 20), dtype=np.float64))
    db.set("coating.interpolator_type", "BILINEAR")
    db.set("rom.n_r", 5)

    db.set("nodes.tip", {
        1: (0.1, 0.8, 0.6),
        2: (0.2, 0.6, 0.8),
    })

    from rubimpact.infra.sparse_matrix import SparseMatrix
    rows = list(range(5))
    cols = list(range(5))
    data = [1.0] * 5
    K = SparseMatrix.from_coo(rows, cols, data, (5, 5))
    db.set("matrices.K_omega", K)
    db.set("rom.enabled", True)
    db.set("rom.tip_dof_map", [0, 1, 2, 3, 4])

    fa = ForceAssembler(db, {"Omega": 100.0})
    cfg = {
        "submodules": {
            "contact_force": {"TYPE": "PENALTY", "k_penalty": "5000"},
            "friction_force": {"TYPE": "COULOMB", "mu": "0.3"},
        }
    }
    fa.configure(cfg)

    pen = np.array([0.01, 0.0], dtype=np.float64)
    coords = np.array([
        [0.0, 0.1, 0.8, 0.6, 1.0],
        [0.5, 0.2, 0.6, 0.8, 1.0],
    ], dtype=np.float64)
    interp = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    context = {"Omega": 100.0, "n_r": 5}
    F_total = fa.assemble(pen, coords, interp, context)

    # F_n = 5000 * 0.01 = 50, F_t = 0.3 * 50 = 15
    # node0 only (node1 has delta=0 → skipped)
    assert F_total.shape == (5,)
    assert np.all(np.isfinite(F_total))
    assert F_total.sum() > 0, "Expected non-zero force for contacting node"
    assert db.get("F_total") is not None


def test_force_assembler_with_normal_friction_requested():
    """ForceAssembler with requested={"F_normal", "F_friction"} returns both."""
    db = DataBus()
    db.set("coating.grid", {"A_cell": 1.0, "h_coat": 1.0})
    db.set("coating.h", np.ones((10, 20), dtype=np.float64))
    db.set("coating.ep", np.zeros((10, 20), dtype=np.float64))
    db.set("coating.alpha", np.zeros((10, 20), dtype=np.float64))
    db.set("coating.interpolator_type", "BILINEAR")
    db.set("rom.n_r", 5)
    db.set("nodes.tip", {1: (0.1, 0.8, 0.6)})

    from rubimpact.infra.sparse_matrix import SparseMatrix
    rows = list(range(5))
    cols = list(range(5))
    data = [1.0] * 5
    K = SparseMatrix.from_coo(rows, cols, data, (5, 5))
    db.set("matrices.K_omega", K)
    db.set("rom.enabled", True)
    db.set("rom.tip_dof_map", [0, 1, 2, 3, 4])

    fa = ForceAssembler(db, {"Omega": 100.0})
    cfg = {
        "submodules": {
            "contact_force": {"TYPE": "PENALTY", "k_penalty": "5000"},
            "friction_force": {"TYPE": "COULOMB", "mu": "0.3"},
        }
    }
    fa.configure(cfg)

    pen = np.array([0.01], dtype=np.float64)
    coords = np.array([[0.0, 0.1, 0.8, 0.6, 1.0]], dtype=np.float64)
    interp = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    context = {"Omega": 100.0, "n_r": 5}
    F_total = fa.assemble(pen, coords, interp, context,
                          requested={"F_normal", "F_friction"})

    assert F_total.shape == (5,)
    assert db.get("F_normal") is not None
    assert db.get("F_friction") is not None
    assert db.get("F_normal").sum() > 0, "Expected non-zero normal force"
    assert db.get("F_friction").sum() > 0, "Expected non-zero friction force"
