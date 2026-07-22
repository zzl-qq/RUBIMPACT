"""Time-step state management."""
import numpy as np


class StateManager:
    def __init__(self, n_dof: int):
        self.u_n   = np.zeros(n_dof, dtype=np.float64)
        self.u_nm1 = np.zeros(n_dof, dtype=np.float64)
        self.t     = 0.0
        self.step  = 0

    def advance(self, u_new: np.ndarray, dt: float) -> None:
        # In-place assignment avoids 2 allocations per step (3M steps → ~6M allocations saved)
        self.u_nm1[:] = self.u_n
        self.u_n[:]   = u_new
        self.t    += dt
        self.step += 1
