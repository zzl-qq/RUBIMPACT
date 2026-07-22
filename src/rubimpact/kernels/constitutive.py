"""Constitutive JIT kernels -- Plastic Coating Law (PCL) return mapping.

Registered types:
    PLASTIC_COATING_LAW -- 1D scalar return mapping, linear isotropic hardening

Signature: (delta, h_loc, ep_loc, alpha_loc, params, out)
    delta:    float64 -- penetration depth
    h_loc:    float64 -- local remaining coating thickness
    ep_loc:   float64 -- local cumulative plastic strain
    alpha_loc:float64 -- local hardening variable
    params:   (E, Y, K_plas, h_coat) float64
    out:      (4,) float64 -- [sigma, ep_new, alpha_new, dw]

Scalar function -- called once per contacting node.
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def evaluate_pcl(delta, h_loc, ep_loc, alpha_loc, params, out):
    """1D return mapping with linear isotropic hardening.

    Wear depth dw is computed from nominal thickness h_coat.

    Args:
        delta: Penetration depth.
        h_loc: Local remaining coating thickness.
        ep_loc: Local cumulative plastic strain (master state variable).
        alpha_loc: Local hardening variable (= ep_loc for linear isotropic).
        params: (E, Y, K_plas, h_coat)
        out: Pre-allocated output array (4,).

    Writes [sigma_new, ep_new, alpha_new, dw] to out.
    """
    E = params[0]
    Y = params[1]
    K_plas = params[2]
    h_coat = params[3]

    if delta <= 0.0 or h_loc <= 0.0:
        out[0] = 0.0
        out[1] = ep_loc
        out[2] = alpha_loc
        out[3] = 0.0
        return

    delta_eff = min(delta, h_loc)
    delta_eps = delta_eff / h_loc
    sigma_trial = E * delta_eps
    f_trial = sigma_trial - (Y + K_plas * alpha_loc)
    if f_trial <= 0.0:
        out[0] = sigma_trial
        out[1] = ep_loc
        out[2] = alpha_loc
        out[3] = 0.0
        return
    dgamma = f_trial / (E + K_plas)
    out[0] = sigma_trial - E * dgamma
    out[1] = ep_loc + dgamma
    out[2] = alpha_loc + dgamma
    out[3] = (out[1] - ep_loc) * h_coat


components.register("constitutive", "PLASTIC_COATING_LAW",
    KernelSpec(fn=evaluate_pcl, signature="pcl_return_mapping",
               stage="compute_normal_force"))
