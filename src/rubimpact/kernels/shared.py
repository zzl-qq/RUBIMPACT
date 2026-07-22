"""Shared kernels: geometry decomposition and ROM DOF mapping.

These stages are automatically inserted by PipelineFactory after all
force-computation stages.  They belong to NO single module — all
force modules (contact_force, friction_force, aero, inertial, ...)
share them.
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(fastmath=True)
def geometry_decompose(F_n: float, F_t: float,
                       yc: float, zc: float, r_val: float) -> tuple:
    """Decompose normal and tangential forces into Y/Z components.

    F_n (radial) → (F_n * yc/r, F_n * zc/r)
    F_t (tangential) → (-F_t * zc/r, F_t * yc/r)

    Used by ALL force modules that produce radial/tangential forces.
    """
    inv_r = 1.0 / max(r_val, 1e-12)
    Fy_n = F_n * yc * inv_r
    Fz_n = F_n * zc * inv_r
    Fy_t = -F_t * zc * inv_r
    Fz_t = F_t * yc * inv_r
    return Fy_n, Fz_n, Fy_t, Fz_t


@njit(fastmath=True)
def rom_dof_map(F_total, F_normal, F_friction,
                Fy_n: float, Fz_n: float, Fy_t: float, Fz_t: float,
                ky: int, kz: int,
                has_normal: bool, has_friction: bool) -> None:
    """Map per-node Y/Z forces to ROM DOF indices.

    Accumulates normal and friction components into global force arrays.
    """
    n_r = F_total.shape[0]
    if 0 <= ky < n_r:
        if has_normal:
            F_normal[ky] += Fy_n
        if has_friction:
            F_friction[ky] += Fy_t
        F_total[ky] += Fy_n + Fy_t
    if 0 <= kz < n_r:
        if has_normal:
            F_normal[kz] += Fz_n
        if has_friction:
            F_friction[kz] += Fz_t
        F_total[kz] += Fz_n + Fz_t


# Register shared kernels
components.register("shared", "GEOMETRY_DECOMPOSE",
    KernelSpec(fn=geometry_decompose, signature="geometry_decompose",
               stage="geometry_decompose"))
components.register("shared", "ROM_DOF_MAP",
    KernelSpec(fn=rom_dof_map, signature="rom_dof_map",
               stage="rom_dof_map"))
