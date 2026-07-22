"""Interpolator JIT kernels — coating state at tip node position.

Registered types:
    BILINEAR — bilinear interpolation on uniform (theta, x) grid
    BICUBIC_BSPLINE — bicubic uniform B-spline interpolation (C^2 continuous)

Signature: (coords, coating_h, coating_ep, coating_alpha, grid_params, out)
    coords:       (n_nodes, 5) float64 — from kinematics [theta, x, yc, zc, r]
    coating_h:    (n_theta, n_x) float64
    coating_ep:   (n_theta, n_x) float64
    coating_alpha:(n_theta, n_x) float64
    grid_params:  (dtheta, dx, x_min, n_theta, n_x, has_coating)
    BILINEAR out: (n_nodes, 11) float64 — [h, ep, alpha, i_t, i_x, w00, w10, w01, w11, gap_pre_h, gap_pre_r]
    BICUBIC_BSPLINE out: (n_nodes, 21) float64 — [h, ep, alpha, i_theta, i_x, w00..w33]
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec

TWO_PI = 2.0 * np.pi


@njit(cache=True, fastmath=True)
def bilinear_interp_kernel(coords, coating_h, coating_ep, coating_alpha,
                            grid_params, out):
    """Bilinear interpolation of coating state at each tip node.

    out columns: 0=h_loc, 1=ep_loc, 2=alpha_loc, 3=i_theta, 4=i_x,
                 5=w00, 6=w10, 7=w01, 8=w11
    """
    dtheta = grid_params[0]
    dx = grid_params[1]
    x_min = grid_params[2]
    n_theta = int(grid_params[3])
    n_x = int(grid_params[4])
    has_coating = grid_params[5] > 0.5

    if not has_coating:
        out[:, 0] = 0.0
        out[:, 1] = 0.0
        out[:, 2] = 0.0
        out[:, 3] = 0.0
        out[:, 4] = 0.0
        out[:, 5] = 1.0
        out[:, 6] = 0.0
        out[:, 7] = 0.0
        out[:, 8] = 0.0
        return

    n_nodes = coords.shape[0]
    for i in range(n_nodes):
        theta_val = coords[i, 0] - np.floor(coords[i, 0] / TWO_PI) * TWO_PI
        x_val = coords[i, 1]
        i_theta = int(np.floor(theta_val / dtheta)) % n_theta
        i_x = int(np.floor((x_val - x_min) / dx))
        i_x = max(0, min(i_x, n_x - 2))
        a = (theta_val - i_theta * dtheta) / dtheta
        b = max(0.0, min(1.0, (x_val - x_min - i_x * dx) / dx))
        i1 = (i_theta + 1) % n_theta
        w00 = (1.0 - a) * (1.0 - b)
        w10 = a * (1.0 - b)
        w01 = (1.0 - a) * b
        w11 = a * b
        out[i, 0] = (w00 * coating_h[i_theta, i_x]
                     + w10 * coating_h[i1, i_x]
                     + w01 * coating_h[i_theta, i_x + 1]
                     + w11 * coating_h[i1, i_x + 1])
        out[i, 1] = (w00 * coating_ep[i_theta, i_x]
                     + w10 * coating_ep[i1, i_x]
                     + w01 * coating_ep[i_theta, i_x + 1]
                     + w11 * coating_ep[i1, i_x + 1])
        out[i, 2] = (w00 * coating_alpha[i_theta, i_x]
                     + w10 * coating_alpha[i1, i_x]
                     + w01 * coating_alpha[i_theta, i_x + 1]
                     + w11 * coating_alpha[i1, i_x + 1])
        out[i, 3] = float(i_theta)
        out[i, 4] = float(i_x)
        out[i, 5] = w00
        out[i, 6] = w10
        out[i, 7] = w01
        out[i, 8] = w11


components.register("interpolator", "BILINEAR",
    KernelSpec(fn=bilinear_interp_kernel, signature="bilinear_interp",
               stage="interpolate_coating"))


@njit(cache=True, fastmath=True)
def bicubic_bspline_interp_kernel(coords, coating_h, coating_ep, coating_alpha,
                                    grid_params, out):
    """Bicubic uniform B-spline interpolation of coating state at each tip node.

    out columns: 0=h_loc, 1=ep_loc, 2=alpha_loc, 3=i_theta, 4=i_x,
                 5=w00, 6=w10, 7=w20, 8=w30,
                 9=w01, 10=w11, 11=w21, 12=w31,
                 13=w02, 14=w12, 15=w22, 16=w32,
                 17=w03, 18=w13, 19=w23, 20=w33
    """
    dtheta = grid_params[0]
    dx = grid_params[1]
    x_min = grid_params[2]
    n_theta = int(grid_params[3])
    n_x = int(grid_params[4])
    has_coating = grid_params[5] > 0.5

    if not has_coating:
        out[:, 0] = 0.0
        out[:, 1] = 0.0
        out[:, 2] = 0.0
        out[:, 3] = 0.0
        out[:, 4] = 0.0
        out[:, 5] = 1.0
        for c in range(6, 21):
            out[:, c] = 0.0
        return

    inv6 = 1.0 / 6.0
    n_nodes = coords.shape[0]

    for i in range(n_nodes):
        theta_val = coords[i, 0] - np.floor(coords[i, 0] / TWO_PI) * TWO_PI
        x_val = coords[i, 1]

        i_theta = int(np.floor(theta_val / dtheta)) % n_theta
        i_x = int(np.floor((x_val - x_min) / dx))
        i_x = max(0, min(i_x, n_x - 4))

        a = (theta_val - float(i_theta) * dtheta) / dtheta
        b = max(0.0, min(1.0, (x_val - x_min - float(i_x) * dx) / dx))

        # Cubic B-spline basis in theta
        a2 = a * a
        a3 = a2 * a
        Bt0 = inv6 * (1.0 - a) * (1.0 - a) * (1.0 - a)
        Bt1 = inv6 * (3.0 * a3 - 6.0 * a2 + 4.0)
        Bt2 = inv6 * (-3.0 * a3 + 3.0 * a2 + 3.0 * a + 1.0)
        Bt3 = inv6 * a3

        # Cubic B-spline basis in x
        b2 = b * b
        b3 = b2 * b
        Bx0 = inv6 * (1.0 - b) * (1.0 - b) * (1.0 - b)
        Bx1 = inv6 * (3.0 * b3 - 6.0 * b2 + 4.0)
        Bx2 = inv6 * (-3.0 * b3 + 3.0 * b2 + 3.0 * b + 1.0)
        Bx3 = inv6 * b3

        # Tensor product weights (theta-major order)
        w00 = Bt0 * Bx0; w10 = Bt1 * Bx0; w20 = Bt2 * Bx0; w30 = Bt3 * Bx0
        w01 = Bt0 * Bx1; w11 = Bt1 * Bx1; w21 = Bt2 * Bx1; w31 = Bt3 * Bx1
        w02 = Bt0 * Bx2; w12 = Bt1 * Bx2; w22 = Bt2 * Bx2; w32 = Bt3 * Bx2
        w03 = Bt0 * Bx3; w13 = Bt1 * Bx3; w23 = Bt2 * Bx3; w33 = Bt3 * Bx3

        # Theta indices with periodic wrap
        it0 = i_theta
        it1 = (i_theta + 1) % n_theta
        it2 = (i_theta + 2) % n_theta
        it3 = (i_theta + 3) % n_theta

        # x indices
        ix0 = i_x
        ix1 = i_x + 1
        ix2 = i_x + 2
        ix3 = i_x + 3

        # Interpolate h
        out[i, 0] = (
            w00 * coating_h[it0, ix0] + w10 * coating_h[it1, ix0]
            + w20 * coating_h[it2, ix0] + w30 * coating_h[it3, ix0]
            + w01 * coating_h[it0, ix1] + w11 * coating_h[it1, ix1]
            + w21 * coating_h[it2, ix1] + w31 * coating_h[it3, ix1]
            + w02 * coating_h[it0, ix2] + w12 * coating_h[it1, ix2]
            + w22 * coating_h[it2, ix2] + w32 * coating_h[it3, ix2]
            + w03 * coating_h[it0, ix3] + w13 * coating_h[it1, ix3]
            + w23 * coating_h[it2, ix3] + w33 * coating_h[it3, ix3])

        # Interpolate ep
        out[i, 1] = (
            w00 * coating_ep[it0, ix0] + w10 * coating_ep[it1, ix0]
            + w20 * coating_ep[it2, ix0] + w30 * coating_ep[it3, ix0]
            + w01 * coating_ep[it0, ix1] + w11 * coating_ep[it1, ix1]
            + w21 * coating_ep[it2, ix1] + w31 * coating_ep[it3, ix1]
            + w02 * coating_ep[it0, ix2] + w12 * coating_ep[it1, ix2]
            + w22 * coating_ep[it2, ix2] + w32 * coating_ep[it3, ix2]
            + w03 * coating_ep[it0, ix3] + w13 * coating_ep[it1, ix3]
            + w23 * coating_ep[it2, ix3] + w33 * coating_ep[it3, ix3])

        # Interpolate alpha
        out[i, 2] = (
            w00 * coating_alpha[it0, ix0] + w10 * coating_alpha[it1, ix0]
            + w20 * coating_alpha[it2, ix0] + w30 * coating_alpha[it3, ix0]
            + w01 * coating_alpha[it0, ix1] + w11 * coating_alpha[it1, ix1]
            + w21 * coating_alpha[it2, ix1] + w31 * coating_alpha[it3, ix1]
            + w02 * coating_alpha[it0, ix2] + w12 * coating_alpha[it1, ix2]
            + w22 * coating_alpha[it2, ix2] + w32 * coating_alpha[it3, ix2]
            + w03 * coating_alpha[it0, ix3] + w13 * coating_alpha[it1, ix3]
            + w23 * coating_alpha[it2, ix3] + w33 * coating_alpha[it3, ix3])

        # Store indices and all 16 weights
        out[i, 3] = float(i_theta)
        out[i, 4] = float(i_x)
        out[i, 5] = w00; out[i, 6] = w10; out[i, 7] = w20; out[i, 8] = w30
        out[i, 9] = w01; out[i, 10] = w11; out[i, 11] = w21; out[i, 12] = w31
        out[i, 13] = w02; out[i, 14] = w12; out[i, 15] = w22; out[i, 16] = w32
        out[i, 17] = w03; out[i, 18] = w13; out[i, 19] = w23; out[i, 20] = w33


components.register("interpolator", "BICUBIC_BSPLINE",
    KernelSpec(fn=bicubic_bspline_interp_kernel, signature="bicubic_bspline_interp",
               stage="interpolate_coating"))
