"""Tests for protocolized TimeIntegrator module."""
import numpy as np
import pytest
from rubimpact.infra.databus import DataBus
from rubimpact.modules.time_integrator import TimeIntegrator, _build_time_integrator
from rubimpact.core.registry import components, KernelSpec, ModuleSpec


# ── Dummy JIT kernels for testing ──

def _dummy_predict_fn(u_n, u_nm1, M_r, K_r, D_r, h, h2, A_inv, out):
    """Dummy predictor: out = u_n."""
    out[:] = u_n


def _dummy_predict_precomputed_fn(u_n, u_nm1, coeff_n, coeff_nm1, A_inv, out):
    """Dummy precomputed predictor: out = u_n."""
    out[:] = u_n


def _dummy_correct_fn(u_p, F_total, A_inv, out):
    """Dummy corrector: out = u_p - A_inv @ F_total."""
    out[:] = u_p - A_inv @ F_total


# ── Fixtures ──

@pytest.fixture(autouse=True)
def _register_dummy_kernels():
    """Register dummy predictor/corrector kernels for testing."""
    components.register("predictor", "LINEAR",
        KernelSpec(fn=_dummy_predict_fn, signature="dummy_predict", stage="predict"))
    components.register("predictor", "LINEAR_PRECOMPUTED",
        KernelSpec(fn=_dummy_predict_precomputed_fn, signature="dummy_predict_precomputed", stage="predict"))
    components.register("predictor", "NONLINEAR",
        KernelSpec(fn=_dummy_predict_fn, signature="dummy_predict", stage="predict"))
    components.register("corrector", "CONTACT_CONSTRAINED",
        KernelSpec(fn=_dummy_correct_fn, signature="dummy_correct", stage="correct"))
    yield
    # No teardown needed — tests are isolated


@pytest.fixture
def rom_db():
    """DataBus with minimal ROM matrices."""
    n_r = 3
    db = DataBus()
    db.set("rom.M_r", np.eye(n_r, dtype=np.float64))
    db.set("rom.K_r", np.zeros((n_r, n_r), dtype=np.float64))
    db.set("rom.D_r", np.zeros((n_r, n_r), dtype=np.float64))
    return db


# ── Registration tests ──

def test_time_integrator_registered():
    """TIME_INTEGRATOR is registered as ModuleSpec."""
    spec = components.resolve_module("TIME_INTEGRATOR", "CENTRAL_DIFFERENCE")
    assert spec is not None
    assert spec.protocol == "TimeIntegratorProtocol"


def test_time_integrator_class_registered():
    """TIME_INTEGRATOR class is registered via register_class."""
    cls = components.resolve_class("TIME_INTEGRATOR")
    assert cls is TimeIntegrator


# ── Protocol tests ──

def test_protocol_has_predict_and_correct_stages(rom_db):
    """Protocol declares predict and correct stages."""
    cfg = {
        "submodules": {
            "predictor": {"TYPE": "LINEAR"},
            "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
        }
    }
    mod = TimeIntegrator(rom_db, {"time_cfg": cfg})
    mod.configure(cfg)
    mod.initialize(h=1e-6)

    proto = mod.get_pipeline_protocol()
    assert proto is not None
    stage_names = [s.name for s in proto.stages]
    assert stage_names == ["predict", "correct"]


def test_protocol_kernel_refs_resolved(rom_db):
    """Protocol kernel_refs point to registered kernels."""
    cfg = {
        "submodules": {
            "predictor": {"TYPE": "LINEAR"},
            "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
        }
    }
    mod = TimeIntegrator(rom_db, {"time_cfg": cfg})
    mod.configure(cfg)
    mod.initialize(h=1e-6)

    proto = mod.get_pipeline_protocol()
    refs = [s.kernel_ref for s in proto.stages]
    # LINEAR auto-upgrades to LINEAR_PRECOMPUTED when available
    assert refs[0] == "predictor/LINEAR_PRECOMPUTED"
    assert refs[1] == "corrector/CONTACT_CONSTRAINED"


def test_protocol_depends_on(rom_db):
    """correct stage depends on predict."""
    cfg = {
        "submodules": {
            "predictor": {"TYPE": "LINEAR"},
            "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
        }
    }
    mod = TimeIntegrator(rom_db, {"time_cfg": cfg})
    mod.configure(cfg)
    mod.initialize(h=1e-6)

    proto = mod.get_pipeline_protocol()
    predict_stage = proto.stages[0]
    correct_stage = proto.stages[1]
    assert predict_stage.depends_on == []
    assert "predict" in correct_stage.depends_on


def test_protocol_params(rom_db):
    """Protocol includes M_r, K_r, D_r, h params."""
    cfg = {
        "submodules": {
            "predictor": {"TYPE": "LINEAR"},
            "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
        }
    }
    mod = TimeIntegrator(rom_db, {"time_cfg": cfg})
    mod.configure(cfg)
    mod.initialize(h=1e-6)

    proto = mod.get_pipeline_protocol()
    assert "M_r" in proto.params
    assert "K_r" in proto.params
    assert "D_r" in proto.params
    assert "h" in proto.params
    assert proto.params["h"] == 1e-6


# ── configure() tests ──

def test_configure_missing_predictor_raises(rom_db):
    """configure raises if no predictor TYPE specified."""
    mod = TimeIntegrator(rom_db)
    with pytest.raises(ValueError, match="predictor"):
        mod.configure({"submodules": {"corrector": {"TYPE": "CONTACT_CONSTRAINED"}}})


def test_configure_missing_corrector_raises(rom_db):
    """configure raises if no corrector TYPE specified."""
    mod = TimeIntegrator(rom_db)
    with pytest.raises(ValueError, match="corrector"):
        mod.configure({"submodules": {"predictor": {"TYPE": "LINEAR"}}})


def test_configure_unknown_predictor_raises(rom_db):
    """configure raises for unknown predictor TYPE."""
    mod = TimeIntegrator(rom_db)
    with pytest.raises(ValueError, match="Unknown predictor TYPE"):
        mod.configure({
            "submodules": {
                "predictor": {"TYPE": "BOGUS"},
                "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
            }
        })


def test_configure_unknown_corrector_raises(rom_db):
    """configure raises for unknown corrector TYPE."""
    mod = TimeIntegrator(rom_db)
    with pytest.raises(ValueError, match="Unknown corrector TYPE"):
        mod.configure({
            "submodules": {
                "predictor": {"TYPE": "LINEAR"},
                "corrector": {"TYPE": "BOGUS"},
            }
        })


# ── Builder tests ──

def test_builder_creates_module(rom_db):
    """Builder function creates and configures a TimeIntegrator."""
    ctx = {
        "time_cfg": {
            "submodules": {
                "predictor": {"TYPE": "LINEAR"},
                "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
            }
        }
    }
    mod = _build_time_integrator(rom_db, ctx)
    assert isinstance(mod, TimeIntegrator)
    mod.initialize(h=1e-6)
    proto = mod.get_pipeline_protocol()
    assert proto is not None


# ── predict/correct functional tests ──

def test_predict_returns_buffer(rom_db):
    """predict uses dummy kernel to compute predicted displacement."""
    cfg = {
        "submodules": {
            "predictor": {"TYPE": "LINEAR"},
            "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
        }
    }
    mod = TimeIntegrator(rom_db)
    mod.configure(cfg)
    mod.initialize(h=1e-6)

    n_r = 3
    u_n = np.ones(n_r, dtype=np.float64)
    u_nm1 = np.zeros(n_r, dtype=np.float64)

    up = mod.predict(u_n, u_nm1)
    assert up is not None
    assert up.shape == (n_r,)
    # Dummy predictor: out = u_n, so up should equal u_n
    np.testing.assert_array_almost_equal(up, u_n)


def test_correct_returns_buffer(rom_db):
    """correct uses dummy kernel to compute corrected displacement."""
    cfg = {
        "submodules": {
            "predictor": {"TYPE": "LINEAR"},
            "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
        }
    }
    mod = TimeIntegrator(rom_db)
    mod.configure(cfg)
    mod.initialize(h=1e-6)

    n_r = 3
    u_p = np.ones(n_r, dtype=np.float64)
    F_total = np.zeros(n_r, dtype=np.float64)

    un = mod.correct(u_p, F_total)
    assert un is not None
    assert un.shape == (n_r,)
    # Dummy corrector: out = u_p - A_inv @ F_total
    # with A_inv = h^2 and F_total = 0, un should == u_p
    np.testing.assert_array_almost_equal(un, u_p)


def test_full_predict_correct_cycle(rom_db):
    """End-to-end predict + correct cycle produces valid results."""
    cfg = {
        "submodules": {
            "predictor": {"TYPE": "LINEAR"},
            "corrector": {"TYPE": "CONTACT_CONSTRAINED"},
        }
    }
    mod = TimeIntegrator(rom_db)
    mod.configure(cfg)
    mod.initialize(h=1e-6)

    n_r = 3
    u_n = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    u_nm1 = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    F_total = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    up = mod.predict(u_n, u_nm1)
    un = mod.correct(up, F_total)

    # Dummy corrector: un = up - A_inv @ F_total
    # with M_r = I, D_r = 0, h = 1e-6
    # A_inv = h^2 * I = 1e-12 * I
    # up = u_n, so un = u_n - 1e-12 * F_total
    expected_un = u_n - 1e-12 * F_total
    np.testing.assert_array_almost_equal(un, expected_un)


# ── Module base class tests ──

def test_module_context_stored():
    """context is stored on the module."""
    ctx = {"key": "value"}
    mod = TimeIntegrator(None, ctx)
    assert mod.ctx == ctx


def test_module_default_context():
    """context defaults to empty dict."""
    mod = TimeIntegrator(None)
    assert mod.ctx == {}
