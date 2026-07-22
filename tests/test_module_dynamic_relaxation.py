"""Tests for protocolized DynamicRelaxation module."""
import numpy as np
import pytest
from rubimpact.infra.databus import DataBus
from rubimpact.modules.dynamic_relaxation import DynamicRelaxation
from rubimpact.core.registry import components, ModuleSpec


# ── Registration tests ──

def test_dr_registered():
    """DYNAMIC_RELAXATION / JACOBI_PRECONDITIONED 已注册。"""
    spec = components.resolve_module("DYNAMIC_RELAXATION", "JACOBI_PRECONDITIONED")
    assert spec is not None
    assert spec.protocol == "DynamicRelaxationProtocol"
    assert isinstance(spec, ModuleSpec)


def test_dr_builder_creates_module():
    """Builder 创建 DynamicRelaxation 实例。"""
    spec = components.resolve_module("DYNAMIC_RELAXATION", "JACOBI_PRECONDITIONED")
    assert spec is not None
    db = DataBus()
    ctx = {"dr_cfg": {"max_steps": 100, "tolerance": 1e-10, "relaxation": 0.5, "force_tol": 1e-6}}
    mod = spec.builder(db, ctx)
    assert isinstance(mod, DynamicRelaxation)


# ── Configuration tests ──

def test_configure_missing_params_raises():
    """缺少必需参数时抛出 ValueError。"""
    db = DataBus()
    mod = DynamicRelaxation(db, {})
    with pytest.raises(ValueError, match="max_steps"):
        mod.configure({})


def test_configure_custom():
    """自定义配置被正确解析。"""
    db = DataBus()
    mod = DynamicRelaxation(db, {})
    mod.configure({
        "max_steps": 100,
        "tolerance": 1e-6,
        "relaxation": 0.3,
        "force_tol": 1e-4,
    })
    assert mod.max_steps == 100
    assert mod.tol == 1e-6
    assert mod.beta == 0.3
    assert mod.force_tol == 1e-4


def test_configure_clamps_beta():
    """beta 被钳位到 [0.01, 1.0]。"""
    base_cfg = {"max_steps": 100, "tolerance": 1e-6, "relaxation": 0.5, "force_tol": 1e-4}
    db = DataBus()
    mod = DynamicRelaxation(db, {})
    mod.configure({**base_cfg, "relaxation": 0.001})
    assert mod.beta == 0.01

    mod2 = DynamicRelaxation(db, {})
    mod2.configure({**base_cfg, "relaxation": 2.0})
    assert mod2.beta == 1.0


def test_builder_integrates_config():
    """通过 builder 的完整构造链。"""
    spec = components.resolve_module("DYNAMIC_RELAXATION", "JACOBI_PRECONDITIONED")
    assert spec is not None
    db = DataBus()
    ctx = {"dr_cfg": {"max_steps": 50, "tolerance": 1e-8, "relaxation": 0.7, "force_tol": 1e-5}}
    mod = spec.builder(db, ctx)
    assert mod.max_steps == 50
    assert mod.tol == 1e-8
    assert mod.beta == 0.7
    assert mod.force_tol == 1e-5


# ── Runtime tests ──

def test_run_converges_on_linear_system():
    """线性系统的 DR 收敛。"""
    db = DataBus()
    mod = DynamicRelaxation(db, {})
    mod.configure({"max_steps": 5000, "tolerance": 1e-10, "relaxation": 0.5, "force_tol": 1e-6})

    n_r = 6
    K_r = np.eye(n_r, dtype=np.float64) * 200.0
    # Apply a small known load: RHS = K * u_true
    u_true = np.array([0.1, -0.05, 0.08, 0.0, 0.02, -0.01], dtype=np.float64)
    F_ext = K_r @ u_true  # external force that would produce u_true at equilibrium

    # Pipeline functions: no contact, just return the external load
    def detect_pipeline(u, t, Omega):
        """Return empty contact data (no contact)."""
        return (np.array(0.0, dtype=np.float64), None, None)

    def assemble_pipeline(pen, coords, interp, Omega):
        """Return -F_ext (CD sign convention: F_contact = -F_ext)."""
        # CD sign convention: assemble returns -F_contact.
        # To get equilibrium K*u = F_ext, we set assemble = -F_ext
        # so that R = -(-F_ext) - K*u = F_ext - K*u.
        return -F_ext

    u, converged, stats = mod.run(
        detect_pipeline, assemble_pipeline,
        u0=None, K_r=K_r, Omega=0.0,
    )

    assert converged
    np.testing.assert_allclose(u, u_true, atol=1e-6)
    assert stats["converged"]


def test_run_handles_u0():
    """通过 u0 设置非零初始位移。"""
    db = DataBus()
    mod = DynamicRelaxation(db, {})
    mod.configure({"max_steps": 200, "tolerance": 1e-12, "relaxation": 0.5, "force_tol": 1e-8})

    n_r = 3
    K_r = np.eye(n_r) * 100.0
    u_eq = np.array([0.2, -0.1, 0.05])

    def detect_pipeline(u, t, Omega):
        return (np.array(0.0), None, None)

    def assemble_pipeline(pen, coords, interp, Omega):
        return -K_r @ u_eq  # F_ext = K * u_eq

    u, converged, stats = mod.run(
        detect_pipeline, assemble_pipeline,
        u0=u_eq, K_r=K_r, Omega=0.0,
    )
    # Already at equilibrium — should converge immediately
    assert converged
    np.testing.assert_allclose(u, u_eq, atol=1e-6)


def test_run_with_coating_protection():
    """涂层在 DR 过程中被保护。"""
    db = DataBus()
    mod = DynamicRelaxation(db, {})
    mod.configure({"max_steps": 200, "tolerance": 1e-10, "relaxation": 0.5, "force_tol": 1e-6})

    n_r = 2
    K_r = np.eye(n_r) * 100.0

    n_coat = 10
    coating_h = np.ones(n_coat, dtype=np.float64) * 0.01
    coating_ep = np.zeros(n_coat, dtype=np.float64)
    coating_alpha = np.zeros(n_coat, dtype=np.float64)

    # Simulate a pipeline that tries to wear the coating each call
    call_count = [0]  # use list to allow mutation in closure

    def detect_pipeline(u, t, Omega):
        return (np.array(0.0), None, None)

    def assemble_pipeline(pen, coords, interp, Omega):
        call_count[0] += 1
        # Simulate wear by modifying coating arrays
        if coating_h is not None and len(coating_h) > 0:
            coating_h[0] -= 0.001
            coating_ep[0] += 0.01
            coating_alpha[0] += 0.005
        return np.zeros(n_r)  # F_total = 0 -> equilibrium at u=0

    u, converged, stats = mod.run(
        detect_pipeline, assemble_pipeline,
        u0=None, K_r=K_r, Omega=0.0,
        coating_h=coating_h, coating_ep=coating_ep, coating_alpha=coating_alpha,
    )

    assert converged
    # After DR, post-convergence wear is applied ONCE
    # coating_h should be modified exactly once (post-convergence)
    assert call_count[0] >= 1
    # The coating should have been restored before each DR iteration,
    # so the final state should reflect exactly one wear application
    # Each call subtracts 0.001; after restoration, only post-convergence
    # call persists -> coating_h starts at 1.0 * 0.01 = 0.01
    assert coating_h[0] == pytest.approx(0.01 - 0.001)  # one wear applied
    assert coating_ep[0] == pytest.approx(0.01)  # one wear applied
    assert coating_alpha[0] == pytest.approx(0.005)


def test_run_non_convergence():
    """dr 在 max_steps 耗尽但不收敛时返回 converged=False。"""
    db = DataBus()
    mod = DynamicRelaxation(db, {})
    mod.configure({"max_steps": 5, "tolerance": 1e-15, "relaxation": 0.5, "force_tol": 1e-15})

    n_r = 2
    K_r = np.eye(n_r) * 100.0
    # Large imbalance that won't converge in 5 steps with tight tolerance
    F_ext = np.array([1e6, -1e6])

    def detect_pipeline(u, t, Omega):
        return (np.array(0.0), None, None)

    def assemble_pipeline(pen, coords, interp, Omega):
        return -F_ext

    u, converged, stats = mod.run(
        detect_pipeline, assemble_pipeline,
        u0=None, K_r=K_r, Omega=0.0,
    )

    assert not converged
    assert stats["converged"] is False
    assert stats["steps"] <= 5
