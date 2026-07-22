"""Predictor JIT kernels -- explicit time integration prediction.

Registered types:
    LINEAR -- central difference predictor: u_p = A_inv @ b_n
    LINEAR_PRECOMPUTED -- same as LINEAR but with pre-computed coefficient matrices

Signature (LINEAR): (u_n, u_nm1, M_r, K_r, D_r, h, h2, A_inv, out)
Signature (LINEAR_PRECOMPUTED): (u_n, u_nm1, coeff_n, coeff_nm1, A_inv, out)
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def linear_predict_kernel(u_n, u_nm1, M_r, K_r, D_r, h, h2, A_inv, out):
    """Central difference explicit predictor.

    coeff_n = 2*M_r/h2 - K_r
    coeff_nm1 = D_r/(2*h) - M_r/h2
    u_p = A_inv @ (coeff_n @ u_n + coeff_nm1 @ u_nm1)
    """
    coeff_n = 2.0 * M_r / h2 - K_r
    coeff_nm1 = D_r / (2.0 * h) - M_r / h2
    b_n = coeff_n @ u_n + coeff_nm1 @ u_nm1
    out[:] = A_inv @ b_n


@njit(cache=True, fastmath=True)
def linear_predict_precomputed_kernel(u_n, u_nm1, coeff_n, coeff_nm1,
                                       A_inv, out):
    """Central difference predictor with pre-computed coefficient matrices.

    coeff_n and coeff_nm1 are constant for the entire simulation (they
    depend only on h, which is fixed).  Pre-computing them in the Python
    layer avoids recomputing element-wise matrix ops every step.
    """
    b_n = coeff_n @ u_n + coeff_nm1 @ u_nm1
    out[:] = A_inv @ b_n


components.register("predictor", "LINEAR",
    KernelSpec(fn=linear_predict_kernel, signature="linear_predict",
               stage="predict"))
components.register("predictor", "LINEAR_PRECOMPUTED",
    KernelSpec(fn=linear_predict_precomputed_kernel, signature="linear_predict_precomputed",
               stage="predict"))
