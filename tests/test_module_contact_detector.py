"""Tests for the protocolized ContactDetector module."""
import numpy as np
import pytest
from rubimpact.infra.databus import DataBus
from rubimpact.core.registry import components
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol
from rubimpact.modules.contact_detector import ContactDetector, tip_dof_indices

# Import kernel modules to trigger ComponentRegistry registration
import rubimpact.kernels.kinematics  # noqa: F401
import rubimpact.kernels.interpolator  # noqa: F401
import rubimpact.kernels.gap  # noqa: F401
import rubimpact.kernels.casing  # noqa: F401


class TestContactDetectorRegistration:
    """ContactDetector is registered as a module class."""

    def test_registered_as_module_class(self):
        """ContactDetector is registered in ComponentRegistry.resolve_class."""
        cls = components.resolve_class("CONTACT_DETECTOR")
        assert cls is not None
        assert cls is ContactDetector

    def test_is_module_subclass(self):
        """ContactDetector is a subclass of Module."""
        assert issubclass(ContactDetector, Module)


class TestPipelineProtocol:
    """get_pipeline_protocol() returns correct 4-stage structure."""

    def test_returns_pipeline_protocol_instance(self):
        """get_pipeline_protocol() returns a PipelineProtocol instance."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        assert isinstance(proto, PipelineProtocol)

    def test_four_stages(self):
        """Pipeline protocol has exactly 4 stages."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        assert len(proto.stages) == 4

    def test_stage_names(self):
        """Stages have the correct names."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        names = [s.name for s in proto.stages]
        assert names == ["compute_kinematics", "compute_casing_radius",
                         "interpolate_coating", "compute_gap"]

    def test_stage_kernel_refs(self):
        """Each stage has a non-None kernel_ref."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        for s in proto.stages:
            assert s.kernel_ref is not None, f"Stage {s.name} has no kernel_ref"
        assert proto.stages[0].kernel_ref.endswith("/" + mod._kinematics_type)
        assert proto.stages[1].kernel_ref == "casing_radius/RGRID_BILINEAR"
        assert proto.stages[2].kernel_ref.endswith("/" + mod._interpolator_type)
        assert proto.stages[3].kernel_ref.endswith("/" + mod._gap_type)

    def test_kinematics_stage_no_dependencies(self):
        """compute_kinematics stage has no dependencies."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        assert proto.stages[0].depends_on == []

    def test_casing_radius_depends_on_kinematics(self):
        """compute_casing_radius depends on compute_kinematics."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        assert proto.stages[1].depends_on == ["compute_kinematics"]

    def test_interpolate_coating_depends_on_kinematics(self):
        """interpolate_coating depends on compute_kinematics."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        assert proto.stages[2].depends_on == ["compute_kinematics"]

    def test_compute_gap_has_two_dependencies(self):
        """compute_gap depends on both compute_casing_radius and interpolate_coating."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        assert proto.stages[3].depends_on == ["compute_casing_radius", "interpolate_coating"]

    def test_params_include_expected_keys(self):
        """Protocol params include theta0, x0, y0, z0, dof_idx, grid_params etc."""
        mod = ContactDetector(None, {})
        proto = mod.get_pipeline_protocol()
        assert "theta0" in proto.params
        assert "x0" in proto.params
        assert "y0" in proto.params
        assert "z0" in proto.params
        assert "dof_idx" in proto.params
        assert "grid_params" in proto.params
        assert "R_grid" in proto.params
        assert "R_grid_params" in proto.params


class TestConfigure:
    """configure() resolves kernels and initializes buffers."""

    def test_configure_sets_kinematics_fn(self):
        """After configure, kinematics kernel is resolved."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)
        assert mod._kinematics_fn is not None

    def test_configure_sets_interpolate_fn(self):
        """After configure, interpolator kernel is resolved."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)
        assert mod._interpolate_fn is not None

    def test_configure_sets_gap_fn(self):
        """After configure, gap function kernel is resolved."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)
        assert mod._gap_fn is not None


class TestDetectSmoke:
    """detect() still works with a minimal DataBus setup."""

    def test_detect_returns_expected_shapes(self):
        """detect() returns pen, coords, interp with correct shapes."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)

        n_r = 6
        u_p = np.zeros(n_r, dtype=np.float64)
        t = 0.0
        Omega = 100.0

        pen, coords, interp = mod.detect(u_p, t, Omega)

        n_nodes = 2
        assert pen.shape == (n_nodes,)
        assert coords.shape == (n_nodes, 5)
        assert interp.shape == (n_nodes, 9)  # BILINEAR = 9 columns
        assert np.all(np.isfinite(pen))
        assert np.all(np.isfinite(coords))
        assert np.all(np.isfinite(interp))

    def test_detect_with_displacement(self):
        """detect() with non-zero displacement produces finite results."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)

        n_r = 6
        u_p = np.array([0.001, 0.0, 0.002, 0.0, 0.0, -0.001], dtype=np.float64)
        t = 0.01
        Omega = 100.0

        pen, coords, interp = mod.detect(u_p, t, Omega)
        assert np.all(np.isfinite(pen))

    def test_penetration_at_nodes_returns_array(self):
        """penetration_at_nodes() works correctly."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)

        n_r = 6
        u_p = np.zeros(n_r, dtype=np.float64)
        pen = mod.penetration_at_nodes(u_p, 0.0, 100.0)
        assert pen.shape == (2,)
        assert np.all(np.isfinite(pen))

    def test_contacts_as_list_returns_list(self):
        """contacts_as_list() returns a list of dicts."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)

        n_r = 6
        u_p = np.zeros(n_r, dtype=np.float64)
        contacts = mod.contacts_as_list(u_p, 0.0, 100.0)
        assert isinstance(contacts, list)

    def test_snapshot_for_output(self):
        """snapshot_for_output() writes to DataBus."""
        db = _make_minimal_db()
        mod = ContactDetector(db)
        cfg = {
            "submodules": {
                "kinematics": {"TYPE": "RIGID_ROTATION_PLUS_VIBRATION"},
                "interpolator": {"TYPE": "BILINEAR"},
                "gap_function": {"TYPE": "DEFAULT"},
            }
        }
        mod.configure(cfg)

        n_r = 6
        u_p = np.zeros(n_r, dtype=np.float64)
        mod.detect(u_p, 0.0, 100.0, copy_to_bus=False)
        mod.snapshot_for_output()
        assert db.get("penetration") is not None
        assert db.get("contacts_coords") is not None
        assert db.get("contacts_interp") is not None


class TestTipDofIndices:
    """Module-level tip_dof_indices() helper function."""

    def test_returns_expected_shape(self):
        """tip_dof_indices returns (n_nodes, 3) int array."""
        db = DataBus()
        db.set("nodes.tip", {
            1: [0.05, 0.2, 0.0],
            2: [0.05, 0.2, 0.3],
        })
        db.set("rom.tip_dof_map", [])
        db.set("rom.enabled", True)
        idx = tip_dof_indices(db)
        assert idx.shape == (2, 3)
        assert idx.dtype == np.int64

    def test_returns_negative_one_for_unmapped(self):
        """Unmapped DOFs get -1 index."""
        db = DataBus()
        db.set("nodes.tip", {
            1: [0.05, 0.2, 0.0],
        })
        db.set("rom.tip_dof_map", [0])  # only DOF 0 mapped
        db.set("rom.enabled", True)
        idx = tip_dof_indices(db)
        assert idx[0, 0] != -1  # DOF 0 mapped
        assert idx[0, 1] == -1  # DOF 1 not mapped
        assert idx[0, 2] == -1  # DOF 2 not mapped


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_minimal_db():
    """Create a minimal DataBus setup for contact detection tests."""
    db = DataBus()

    # Tip nodes: 2 nodes, 3 DOF each
    db.set("nodes.tip", {
        1: [0.05, 0.2, 0.0],
        2: [0.05, 0.2, 0.3],
    })
    db.set("rom.tip_dof_map", [0, 1, 2, 3, 4, 5])
    db.set("rom.enabled", True)

    # Casing geometry (constant radius)
    db.set("casing.geometry", {"get_radius": lambda x, theta: 0.25})

    # Coating grid
    db.set("coating.grid", {
        "dtheta": 0.1,
        "dx": 0.01,
        "x": [0.0, 0.01, 0.02],
        "h_coat": 0.005,
        "A_cell": 0.001,
    })

    # Coating arrays (small grid: n_theta=3, n_x=3)
    coating_h = np.ones((3, 3), dtype=np.float64) * 0.005
    coating_ep = np.zeros((3, 3), dtype=np.float64)
    coating_alpha = np.zeros((3, 3), dtype=np.float64)
    db.set("coating.h", coating_h)
    db.set("coating.ep", coating_ep)
    db.set("coating.alpha", coating_alpha)

    # Casing R_grid for JIT fast path
    R_grid = np.full((3, 3), 0.25, dtype=np.float64)
    db.set("casing.R_grid", R_grid)
    db.set("casing.R_grid_params", np.array([0.1, 0.01, 0.0, 3.0, 3.0], dtype=np.float64))

    return db
