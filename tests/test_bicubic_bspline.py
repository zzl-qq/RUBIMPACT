"""Tests for bicubic B-spline interpolator and wear kernel."""
import numpy as np
from numba import njit


# -- Directly compute B-spline basis in tests (no JIT registry dependency) --

@njit(cache=True)
def _cubic_bspline_test(u):
    """Cubic B-spline basis at u in [0, 1]. Returns (B0, B1, B2, B3)."""
    inv6 = 1.0 / 6.0
    u2 = u * u
    u3 = u2 * u
    return (inv6 * (1.0 - u) * (1.0 - u) * (1.0 - u),
            inv6 * (3.0 * u3 - 6.0 * u2 + 4.0),
            inv6 * (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0),
            inv6 * u3)


def test_cubic_bspline_partition_of_unity():
    """B-spline basis partition of unity: sum B_i(u) = 1."""
    for u in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        B0, B1, B2, B3 = _cubic_bspline_test(u)
        total = B0 + B1 + B2 + B3
        assert abs(total - 1.0) < 1e-14, f"u={u}: sum={total}"


def test_cubic_bspline_symmetry():
    """B_i(u) = B_{3-i}(1-u)."""
    for u in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        b_fwd = _cubic_bspline_test(u)
        b_rev = _cubic_bspline_test(1.0 - u)
        assert abs(b_fwd[0] - b_rev[3]) < 1e-14
        assert abs(b_fwd[1] - b_rev[2]) < 1e-14
        assert abs(b_fwd[2] - b_rev[1]) < 1e-14
        assert abs(b_fwd[3] - b_rev[0]) < 1e-14


def test_interp_constant_field():
    """Constant coating field should be preserved under interpolation."""
    import rubimpact.kernels.interpolator  # trigger registration
    from rubimpact.core.registry import components

    spec = components.resolve_kernel("interpolator", "BICUBIC_BSPLINE")
    assert spec is not None
    kernel = spec.fn
    assert kernel is not None

    n_th, n_x = 6, 8
    h_val, ep_val, alpha_val = 0.5, 0.1, 0.2
    coating_h = np.full((n_th, n_x), h_val, dtype=np.float64)
    coating_ep = np.full((n_th, n_x), ep_val, dtype=np.float64)
    coating_alpha = np.full((n_th, n_x), alpha_val, dtype=np.float64)

    coords = np.zeros((1, 5), dtype=np.float64)
    coords[0, 0] = 1.5  # theta ~ 1.5 rad
    coords[0, 1] = 2.0  # x ~ 2.0 mm

    n_theta_f, n_x_f = float(n_th), float(n_x)
    dtheta = 2.0 * np.pi / n_theta_f
    dx = 1.0  # arbitrary unit length
    grid_params = np.array([dtheta, dx, 0.0, n_theta_f, n_x_f, 1.0], dtype=np.float64)

    out = np.zeros((1, 21), dtype=np.float64)
    kernel(coords, coating_h, coating_ep, coating_alpha, grid_params, out)

    assert abs(out[0, 0] - h_val) < 1e-14, f"h: {out[0, 0]}"
    assert abs(out[0, 1] - ep_val) < 1e-14, f"ep: {out[0, 1]}"
    assert abs(out[0, 2] - alpha_val) < 1e-14, f"alpha: {out[0, 2]}"
    # weight sum = 1
    weight_sum = np.sum(out[0, 5:21])
    assert abs(weight_sum - 1.0) < 1e-14, f"weight sum: {weight_sum}"


def test_weights_partition_unity():
    """Bicubic tensor-product weights sum to 1."""
    for a in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for b in [0.0, 0.3, 0.6, 1.0]:
            Bt = _cubic_bspline_test(a)
            Bx = _cubic_bspline_test(b)
            total = 0.0
            for p in range(4):
                for q in range(4):
                    total += Bt[p] * Bx[q]
            assert abs(total - 1.0) < 1e-14, f"a={a}, b={b}: sum={total}"


def test_wear_bicubic_distribution():
    """磨损应变按 16 权重分布，ep 递增。"""
    import rubimpact.kernels.wear  # trigger registration
    from rubimpact.core.registry import components

    spec = components.resolve_kernel("wear", "PLASTIC_STRAIN_BICUBIC")
    assert spec is not None
    kernel = spec.fn

    n_th, n_x = 6, 8
    h_coat = 0.5
    coating_h = np.full((n_th, n_x), h_coat, dtype=np.float64)
    coating_ep = np.zeros((n_th, n_x), dtype=np.float64)
    coating_alpha = np.zeros((n_th, n_x), dtype=np.float64)

    dgamma = 0.1
    params = np.array([float(n_th), h_coat], dtype=np.float64)

    # 所有权重均等 → 每格点增量 = 0.1 / 16
    w = 1.0 / 16.0
    kernel(0, 0,
           w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w,
           dgamma, coating_h, coating_ep, coating_alpha, params)

    expected_ep = dgamma / 16.0
    for p in range(4):
        for q in range(4):
            jt = p % n_th
            jx = q
            assert abs(coating_ep[jt, jx] - expected_ep) < 1e-14, \
                f"ep[{jt},{jx}]={coating_ep[jt, jx]} != {expected_ep}"
            assert abs(coating_h[jt, jx] - h_coat * (1.0 - expected_ep)) < 1e-14
