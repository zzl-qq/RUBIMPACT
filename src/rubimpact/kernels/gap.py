"""Gap function JIT kernels — penetration detection.

Registered types:
    DEFAULT — g = R_casing - h_loc - r, delta = max(0, -g)

Signature: (coords, coating_interp, R_tip, out)
    coords:         (n_nodes, 5) float64 — from kinematics [theta, x, yc, zc, r]
    coating_interp: (n_nodes, 9) float64 — from interpolator [h, ep, alpha, i_t, i_x, w00, w10, w01, w11]
    R_tip:          (n_nodes,) float64 — pre-computed casing radius per node
    out:            (n_nodes,) float64 — penetration depth delta (0.0 if no contact)
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def default_gap_kernel(coords, coating_interp, R_tip, out):
    """Compute gap g = R - h_loc - r, output penetration delta = max(0, -g)."""
    n_nodes = coords.shape[0]
    for i in range(n_nodes):
        h_loc = coating_interp[i, 0]
        r_val = coords[i, 4]
        R_casing = R_tip[i]
        gap = R_casing - h_loc - r_val
        if gap < 0.0:
            out[i] = -gap
        else:
            out[i] = 0.0


components.register("gap_function", "DEFAULT",
    KernelSpec(fn=default_gap_kernel, signature="default_gap",
               stage="compute_gap"))
