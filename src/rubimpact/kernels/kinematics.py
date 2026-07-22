"""Kinematics JIT kernels — blade tip node motion.

Registered types:
    RIGID_ROTATION_PLUS_VIBRATION — theta = Omega*t + theta0, x = x0 + u_x, r = sqrt(y^2+z^2)

Signature: (u_p, theta0, x0, y0, z0, dof_idx, Omega, t, out)
    u_p:       (n_r,) float64 — predicted displacement in ROM space
    theta0:    (n_nodes,) float64
    x0,y0,z0:  (n_nodes,) float64
    dof_idx:   (n_nodes, 3) int64 — ROM DOF indices per node
    Omega:     float64
    t:         float64
    out:       (n_nodes, 5) float64 — output: [theta, x, yc, zc, r]
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def rigid_rotation_kernel(u_p, theta0, x0, y0, z0, dof_idx, Omega, t, out):
    """Compute tip node positions at time t.

    out columns: 0=theta, 1=x, 2=yc, 3=zc, 4=r
    """
    n_r = u_p.shape[0]
    n_nodes = theta0.shape[0]
    for i in range(n_nodes):
        kx = dof_idx[i, 0]
        ky = dof_idx[i, 1]
        kz = dof_idx[i, 2]
        ux = u_p[kx] if 0 <= kx < n_r else 0.0
        uy = u_p[ky] if 0 <= ky < n_r else 0.0
        uz = u_p[kz] if 0 <= kz < n_r else 0.0
        out[i, 0] = Omega * t + theta0[i]
        out[i, 1] = x0[i] + ux
        out[i, 2] = y0[i] + uy
        out[i, 3] = z0[i] + uz
        out[i, 4] = np.sqrt(out[i, 2] * out[i, 2] + out[i, 3] * out[i, 3])


components.register("kinematics", "RIGID_ROTATION_PLUS_VIBRATION",
    KernelSpec(fn=rigid_rotation_kernel,
               signature="rigid_rotation_plus_vibration",
               stage="compute_kinematics"))
