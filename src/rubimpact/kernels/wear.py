"""Wear JIT kernels -- coating state write-back.

Registered types:
    PLASTIC_STRAIN -- plastic strain-driven wear, bilinear (2x2) weight distribution
    PLASTIC_STRAIN_BICUBIC -- plastic strain-driven wear, bicubic B-spline (4x4) distribution
    NONE -- no wear (no-op)
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def wear_plastic_strain_kernel(i_theta, i_x, w00, w10, w01, w11,
                                dgamma, coating_h, coating_ep, coating_alpha,
                                params):
    """Distribute plastic strain increment to coating grid corners.

    ep accumulates monotonically; h is derived: h = h_coat * (1 - ep).
    """
    n_theta = int(params[0])
    h_coat = params[1]
    i1 = (i_theta + 1) % n_theta

    for jt, jx, w in ((i_theta, i_x, w00),
                      (i1, i_x, w10),
                      (i_theta, i_x + 1, w01),
                      (i1, i_x + 1, w11)):
        dg_corner = dgamma * w
        ep_c = min(max(coating_ep[jt, jx] + dg_corner, 0.0), 1.0)
        coating_ep[jt, jx] = ep_c
        coating_alpha[jt, jx] = ep_c
        coating_h[jt, jx] = h_coat * (1.0 - ep_c)


@njit(cache=True, fastmath=True)
def wear_plastic_strain_bicubic_kernel(i_theta, i_x,
                                         w00, w10, w20, w30,
                                         w01, w11, w21, w31,
                                         w02, w12, w22, w32,
                                         w03, w13, w23, w33,
                                         dgamma, coating_h, coating_ep, coating_alpha,
                                         params):
    """Distribute plastic strain increment to 16-cell (4x4) B-spline stencil.

    ep accumulates monotonically; h is derived: h = h_coat * (1 - ep).
    All nine alpha columns follow ep for isotropic hardening.
    """
    n_theta = int(params[0])
    h_coat = params[1]

    it0 = i_theta % n_theta
    it1 = (i_theta + 1) % n_theta
    it2 = (i_theta + 2) % n_theta
    it3 = (i_theta + 3) % n_theta
    ix0 = i_x; ix1 = i_x + 1; ix2 = i_x + 2; ix3 = i_x + 3

    grid = [(it0, ix0, w00), (it1, ix0, w10), (it2, ix0, w20), (it3, ix0, w30),
            (it0, ix1, w01), (it1, ix1, w11), (it2, ix1, w21), (it3, ix1, w31),
            (it0, ix2, w02), (it1, ix2, w12), (it2, ix2, w22), (it3, ix2, w32),
            (it0, ix3, w03), (it1, ix3, w13), (it2, ix3, w23), (it3, ix3, w33)]

    for jt, jx, w in grid:
        dg_corner = dgamma * w
        ep_c = min(max(coating_ep[jt, jx] + dg_corner, 0.0), 1.0)
        coating_ep[jt, jx] = ep_c
        coating_alpha[jt, jx] = ep_c
        coating_h[jt, jx] = h_coat * (1.0 - ep_c)


@njit(cache=True, fastmath=True)
def wear_none_kernel(i_theta, i_x, w00, w10, w01, w11,
                      dgamma, coating_h, coating_ep, coating_alpha,
                      params):
    """No wear -- no-op."""
    pass


@njit(cache=True, fastmath=True)
def wear_unified_kernel(i_theta, i_x, weights, dgamma,
                         coating_h, coating_ep, coating_alpha, params):
    """Unified wear kernel — dispatches on number of interpolation weights.

    Supports both bilinear (4 weights from 2×2 stencil) and bicubic B-spline
    (16 weights from 4×4 stencil).  The weights are passed as a 1-D array
    slice from the interp buffer row, avoiding Numba signature mismatch
    between the two kernel arities inside the compiled pipeline.

    Parameters
    ----------
    i_theta : int
        Base circumferential index.
    i_x : int
        Base axial index.
    weights : 1-D float64 array
        Interpolation weights, length 4 (bilinear) or 16 (bicubic).
    dgamma : float
        Plastic strain increment to distribute.
    coating_h, coating_ep, coating_alpha : 2-D float64 arrays
        Coating state grids (modified in-place).
    params : 1-D float64 array
        [n_theta, h_coat].
    """
    n_theta = int(params[0])
    h_coat = params[1]
    n_w = weights.shape[0]
    base = 2 if n_w <= 4 else 4

    for c in range(n_w):
        jt = (i_theta + (c % base)) % n_theta
        jx = i_x + (c // base)
        dg_corner = dgamma * weights[c]
        ep_c = min(max(coating_ep[jt, jx] + dg_corner, 0.0), 1.0)
        coating_ep[jt, jx] = ep_c
        coating_alpha[jt, jx] = ep_c
        coating_h[jt, jx] = h_coat * (1.0 - ep_c)

components.register("wear", "PLASTIC_STRAIN",
    KernelSpec(fn=wear_plastic_strain_kernel, signature="plastic_strain_wear",
               stage="apply_wear"))
components.register("wear", "PLASTIC_STRAIN_BICUBIC",
    KernelSpec(fn=wear_plastic_strain_bicubic_kernel, signature="plastic_strain_bicubic_wear",
               stage="apply_wear"))
components.register("wear", "UNIFIED",
    KernelSpec(fn=wear_unified_kernel, signature="wear_unified",
               stage="apply_wear"))
components.register("wear", "NONE",
    KernelSpec(fn=wear_none_kernel, signature="none",
               stage="apply_wear"))
