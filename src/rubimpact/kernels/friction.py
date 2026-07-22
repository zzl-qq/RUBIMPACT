"""Friction JIT kernels -- tangential force models.

Registered types:
    COULOMB -- F_t = mu * F_n
    STRIBECK -- mu(v) = mu_d + (mu_s - mu_d) * exp(-|v|/v_s), F_t = mu(v) * F_n

Signature: (F_n, v_rel, params, out)
    F_n:    float64 -- normal contact force
    v_rel:  float64 -- relative sliding velocity
    params: model-specific (Coulomb: [mu], Stribeck: [mu_s, mu_d, v_s])
    out:    (1,) float64 -- tangential force F_t
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def coulomb_kernel(F_n, v_rel, params, out):
    """Coulomb friction: F_t = mu * F_n."""
    out[0] = params[0] * F_n


@njit(cache=True, fastmath=True)
def stribeck_kernel(F_n, v_rel, params, out):
    """Stribeck friction: mu(v) = mu_d + (mu_s - mu_d) * exp(-|v|/v_s)."""
    mu_s = params[0]
    mu_d = params[1]
    v_s = params[2]
    mu_v = mu_d + (mu_s - mu_d) * np.exp(-abs(v_rel) / v_s)
    out[0] = mu_v * F_n


components.register("friction", "COULOMB",
    KernelSpec(fn=coulomb_kernel, signature="coulomb",
               stage="compute_friction_force"))
components.register("friction", "STRIBECK",
    KernelSpec(fn=stribeck_kernel, signature="stribeck",
               stage="compute_friction_force"))
