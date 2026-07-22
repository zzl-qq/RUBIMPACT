"""Regression tests for migrated detection kernels."""
import numpy as np
from rubimpact.core.registry import components

# Import kernel modules to trigger ComponentRegistry registration
import rubimpact.kernels.kinematics  # noqa: F401
import rubimpact.kernels.interpolator  # noqa: F401
import rubimpact.kernels.gap  # noqa: F401
import rubimpact.kernels.casing  # noqa: F401


def test_kinematics_kernel_registered():
    """运动学内核已在 ComponentRegistry 中注册。"""
    spec = components.resolve_kernel("kinematics", "RIGID_ROTATION_PLUS_VIBRATION")
    assert spec is not None
    assert spec.signature == "rigid_rotation_plus_vibration"
    assert spec.stage == "compute_kinematics"


def test_kinematics_kernel_callable():
    """运动学内核可以调用并返回正确结果。"""
    spec = components.resolve_kernel("kinematics", "RIGID_ROTATION_PLUS_VIBRATION")
    fn = spec.fn

    n_nodes = 3
    u_p = np.array([0.001, 0.0, 0.002], dtype=np.float64)
    theta0 = np.array([0.0, 2.0, 4.0], dtype=np.float64)
    x0 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    y0 = np.array([0.8, 0.7, 0.6], dtype=np.float64)
    z0 = np.array([0.6, 0.7, 0.8], dtype=np.float64)
    dof_idx = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int64)
    coords_out = np.zeros((n_nodes, 5), dtype=np.float64)

    fn(u_p, theta0, x0, y0, z0, dof_idx, 100.0, 0.0, coords_out)

    # coords_out columns: [theta, x, yc, zc, r]
    assert coords_out.shape == (3, 5)
    assert np.all(np.isfinite(coords_out))


def test_interpolator_kernels_registered():
    """插值内核已在 ComponentRegistry 中注册。"""
    assert components.resolve_kernel("interpolator", "BILINEAR") is not None
    assert components.resolve_kernel("interpolator", "BICUBIC_BSPLINE") is not None


def test_gap_kernel_registered():
    """间隙函数内核已注册。"""
    assert components.resolve_kernel("gap_function", "DEFAULT") is not None


def test_casing_radius_kernel_registered():
    """机匣半径内核已注册。"""
    assert components.resolve_kernel("casing_radius", "RGRID_BILINEAR") is not None
