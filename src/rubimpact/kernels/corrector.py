"""Corrector JIT kernels -- displacement correction.

Registered types:
    CONTACT_CONSTRAINED -- u_{n+1} = u_p - A_inv @ F_total

Signature: (u_p, F_total, A_inv, out)
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def contact_constrained_correct_kernel(u_p, F_total, A_inv, out):
    """Contact-constrained correction.

    u_{n+1} = u_p - A_inv @ F_total
    """
    out[:] = u_p - A_inv @ F_total


components.register("corrector", "CONTACT_CONSTRAINED",
    KernelSpec(fn=contact_constrained_correct_kernel, signature="contact_constrained_correct",
               stage="correct"))
