"""Casing radius JIT kernels — bilinear interpolation of pre-computed R_grid.

Registered types:
    RGRID_BILINEAR — bilinear interpolation of casing.R_grid at tip node positions

This replaces the Python loop ``_compute_R_tip()`` in ContactDetector with
a single Numba-compiled call.  The R_grid is pre-computed once at init time
by ``Casing.build_R_grid()``, so the JIT kernel contains zero casing
geometry knowledge — it just bilinearly interpolates a 2D array.

Signature: (coords, R_grid, grid_params, out)
    coords:      (n_nodes, 5) float64 — from kinematics [theta, x, yc, zc, r]
    R_grid:      (n_theta, n_x) float64 — pre-computed casing radius grid
    grid_params: (6,) float64 — [dtheta, dx, x_min, theta_min, n_theta, n_x]
    out:         (n_nodes,) float64 — casing radius at each tip node
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec

TWO_PI = 2.0 * np.pi


@njit(cache=True, fastmath=True)
def rgrid_bilinear_kernel(coords, R_grid, grid_params, out):
    """Bilinear interpolation of pre-computed casing radius grid at each tip node.

    For each node i:
        theta = coords[i, 0] mod 2π
        x = coords[i, 1]
        Bilinear interpolation of R_grid at (theta, x) → out[i]

    This is structurally identical to the coating bilinear interpolator but
    only interpolates a single field (casing radius) instead of three fields.
    """
    dtheta = grid_params[0]
    dx = grid_params[1]
    x_min = grid_params[2]
    theta_min = grid_params[3]
    n_theta = int(grid_params[4])
    n_x = int(grid_params[5])

    n_nodes = coords.shape[0]
    for i in range(n_nodes):
        # Normalise theta to [theta_min, theta_min+2π)
        theta_val = coords[i, 0] - np.floor(
            (coords[i, 0] - theta_min) / TWO_PI) * TWO_PI
        x_val = coords[i, 1]

        i_theta = int(np.floor((theta_val - theta_min) / dtheta)) % n_theta
        i_x = int(np.floor((x_val - x_min) / dx))
        i_x = max(0, min(i_x, n_x - 2))

        a = (theta_val - (theta_min + i_theta * dtheta)) / dtheta
        b = max(0.0, min(1.0, (x_val - x_min - i_x * dx) / dx))

        i1 = (i_theta + 1) % n_theta
        w00 = (1.0 - a) * (1.0 - b)
        w10 = a * (1.0 - b)
        w01 = (1.0 - a) * b
        w11 = a * b

        out[i] = (w00 * R_grid[i_theta, i_x]
                  + w10 * R_grid[i1, i_x]
                  + w01 * R_grid[i_theta, i_x + 1]
                  + w11 * R_grid[i1, i_x + 1])


components.register("casing_radius", "RGRID_BILINEAR",
    KernelSpec(fn=rgrid_bilinear_kernel, signature="rg_grid_bilinear",
               stage="compute_casing_radius"))
