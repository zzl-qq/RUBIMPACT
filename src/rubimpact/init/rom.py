"""*ROM module — Craig-Bampton reduction (framework S4.5).

Craig-Bampton (fixed-interface) component mode synthesis:

1. DOF partition: boundary (b) = tip contact DOF, interior (i) = the rest.
   Constrained DOF (penalty diagonal >= 1e30) are removed beforehand.
2. Constraint modes:      Psi_c = -K_ii^{-1} K_ib          (PARDISO)
3. Fixed-interface modes: K_ii phi = w^2 M_ii phi          (shift-invert
   ARPACK, sigma=0, with the PARDISO factorization of K_ii as OPinv;
   modes are M_ii-orthonormal)
4. CB transformation:     Phi_CB = [[I, 0], [Psi_c, Phi_n]]
5. Projection:            X_r = Phi_CB^T X Phi_CB  for X in {M, K, D}
6. ROM eigenvalue check:  K_r psi = w_r^2 M_r psi   (dense eigh, small n_r)
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from rubimpact.infra.databus import DataBus
from rubimpact.core.module_base import Module

DOF_PER_NODE = 3
CONSTRAINED_DIAG = 1e30


class ROM(Module):
    """Craig-Bampton reduced-order model."""

    def configure(self, cfg: dict) -> None:
        n_modal = int(cfg["n_modal"])
        M_full = self.db.get("matrices.mass").csr
        K_full = self.db.get("matrices.K_omega").csr
        D_full = self.db.get("matrices.D_full").csr
        tip_nodes = self.db.get("nodes.tip", {})

        # Step 0: remove constrained DOF (penalty diagonal >= 1e30)
        diag_K = K_full.diagonal()
        free_dofs = np.where(diag_K < CONSTRAINED_DIAG)[0]
        n_free = len(free_dofs)

        # Step 1: DOF partition — boundary = tip contact DOF, interior = rest
        tip_dof_map = self._build_tip_dof_map(tip_nodes, free_dofs)
        if not tip_dof_map:
            raise ValueError(
                "ROM requires tip node coordinates (nodes.tip with free DOF)")
        free_to_idx = {int(d): i for i, d in enumerate(free_dofs)}
        b_local = np.array([free_to_idx[d] for d in tip_dof_map],
                           dtype=np.int64)
        b_set = set(b_local.tolist())
        i_local = np.array([j for j in range(n_free) if j not in b_set],
                           dtype=np.int64)
        n_i, n_b = len(i_local), len(b_local)

        gi = free_dofs[i_local]
        gb = free_dofs[b_local]
        K_ii = K_full[gi][:, gi].tocsr()
        K_ib = K_full[gi][:, gb].tocsr()
        M_ii = M_full[gi][:, gi].tocsr()

        # Step 2: constraint modes Psi_c = -K_ii^{-1} K_ib (PARDISO, multi-RHS)
        from pypardiso import PyPardisoSolver
        solver = PyPardisoSolver()
        solver.factorize(K_ii)
        psi_c = solver.solve(K_ii, -K_ib.toarray())
        psi_c = psi_c.reshape(n_i, n_b)

        # Step 3: fixed-interface normal modes K_ii phi = w^2 M_ii phi.
        n_k = min(n_modal, n_i - 1)
        op_inv = LinearOperator(
            (n_i, n_i), matvec=lambda v: solver.solve(K_ii, v))
        evals, phi_n = eigsh(K_ii, k=n_k, M=M_ii, sigma=0.0, which="LM",
                             OPinv=op_inv)
        n_k = phi_n.shape[1]

        # --- Sanity checks & diagnostic output ---
        if np.any(evals <= 0.0):
            raise ValueError(
                f"K_ii is not positive-definite: {np.sum(evals <= 0)} "
                f"non-positive eigenvalue(s) detected.  Check the FE model "
                f"for unconstrained rigid-body modes or negative Jacobians.")
        print(f"  [ROM] Craig-Bampton reduction: n_modal={n_modal}, "
              f"n_b={n_b}, n_i={n_i}, n_r={n_b + n_k}")

        # Step 4: assemble Phi_CB = [[I, 0], [Psi_c, Phi_n]] (free-DOF rows)
        n_r = n_b + n_k
        Phi_CB = np.zeros((n_free, n_r))
        Phi_CB[b_local, np.arange(n_b)] = 1.0
        Phi_CB[i_local, :n_b] = psi_c
        Phi_CB[i_local, n_b:] = phi_n

        # Step 5: ROM projection X_r = Phi_CB^T X Phi_CB
        M_free = M_full[free_dofs][:, free_dofs]
        K_free = K_full[free_dofs][:, free_dofs]
        D_free = D_full[free_dofs][:, free_dofs]
        M_r = Phi_CB.T @ (M_free @ Phi_CB)
        K_r = Phi_CB.T @ (K_free @ Phi_CB)
        D_r = Phi_CB.T @ (D_free @ Phi_CB)

        # Step 6: solve ROM eigenvalue problem
        from scipy.linalg import eigh
        evals_rom, _ = eigh(K_r, M_r)
        freqs_rom = np.sqrt(np.maximum(evals_rom, 0.0)) / (2.0 * np.pi)
        print(f"  [ROM] reduced system natural frequencies (Hz) — "
              f"n_r={n_r} ({n_b} boundary + {n_k} modal):")
        for j, f in enumerate(freqs_rom):
            print(f"    mode {j + 1:3d}: {f:12.4f} Hz")

        self.db.set("rom.M_r", M_r)
        self.db.set("rom.K_r", K_r)
        self.db.set("rom.D_r", D_r)
        self.db.set("rom.Phi_CB", Phi_CB)
        self.db.set("rom.n_r", n_r)
        self.db.set("rom.enabled", True)
        self.db.set("rom.tip_dof_map", tip_dof_map)
        self.db.set("rom.free_dofs", free_dofs)

    @staticmethod
    def _build_tip_dof_map(tip_nodes: dict, free_dofs: np.ndarray) -> list:
        """Global DOF indices of tip nodes, restricted to free DOF."""
        free_set = set(free_dofs.tolist())
        tip_dofs = []
        for nid in sorted(tip_nodes.keys()):
            base = (nid - 1) * DOF_PER_NODE
            for d in range(DOF_PER_NODE):
                dof = base + d
                if dof in free_set:
                    tip_dofs.append(dof)
        return tip_dofs


# Register for auto-discovery
from rubimpact.core.registry import components as registry
registry.register_class("ROM", ROM)
