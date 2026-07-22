"""Protocolized DynamicRelaxation module — Jacobi-preconditioned static equilibrium.

Converts the standalone `dynamic_relaxation()` function (runtime/) into a
Module subclass that receives PIPELINE FUNCTIONS via `run()` instead of
module instances (ti, cd, fa).  This decouples DR from implementation
details of contact detection and force assembly.

Problem
-------
The original implementation reused the transient central-difference
integrator at h ~ 1e-8 s.  The effective iteration matrix is
A ~ M/h^2 ~ 1e16, making each DR step move only ~1e-16 — requiring
trillions of iterations.

Solution
--------
Use the diagonal of the stiffness matrix (Jacobi preconditioner):

    u_{k+1} = u_k + beta * diag(K)^(-1) * F(u_k)

For each DOF i, the step is beta * F_i / K_ii.  This has three key
advantages over the mass-scaled approach:

1. **Per-DOF scaling**: soft DOFs (small K_ii) get larger steps for
   faster convergence; stiff DOFs get smaller steps for stability.
   No single omega_max bottleneck.

2. **Contact-safe**: if contact adds stiffness deltaK on DOF i, the
   effective step becomes beta/(K_ii + deltaK) < beta/K_ii — inherently
   stable against contact stiffening.

3. **No power iteration needed**: diag(K) is free to compute.

Convergence
-----------
For the linear case F = F_ext - K*u:

    u_{k+1} = (I - beta*D^(-1)*K)*u_k + beta*D^(-1)*F_ext

where D = diag(K).  The spectral radius is typically 0.5-0.9 for
well-conditioned structural problems, giving ~10-100 iterations to
machine precision.

INP keyword
-----------
::

    *DYNAMIC_RELAXATION
      max_steps=10000, tolerance=1e-10, relaxation=0.5
      force_tol=1e-6

- ``max_steps``    max iterations (required)
- ``tolerance``    |du|_inf convergence threshold (required)
- ``relaxation``   beta in (0, 1] — step fraction (required)
- ``force_tol``      |F|_inf equilibrium threshold (required)
"""

import numpy as np
from rubimpact.core.registry import components, ModuleSpec
from rubimpact.core.module_base import Module


class DynamicRelaxation(Module):
    """Jacobi-preconditioned dynamic relaxation for static equilibrium.

    Receives pipeline functions (not module instances) via ``run()``,
    decoupling DR from the internal details of contact detection and
    force assembly.

    Pipeline function signatures
    ----------------------------
    - ``detect_pipeline(u, t, Omega) -> (pen, coords, interp)``
    - ``assemble_pipeline(pen, coords, interp, Omega) -> F_total``

    Coating wear protection
    -----------------------
    Wear is time-dependent — it should not accumulate during the
    iterative DR solve.  The coating arrays (h, ep, alpha) are saved
    before each force evaluation and restored afterward.  Every DR
    iteration thus sees the **initial (unworn)** coating.  After
    convergence, wear is applied **once** from the converged state.
    """

    def configure(self, cfg: dict) -> None:
        """Parse *DYNAMIC_RELAXATION INP config.

        All four parameters are required — no silent defaults.
        Provide them via the *DYNAMIC_RELAXATION keyword in the INP file:

            *DYNAMIC_RELAXATION
              max_steps=10000, tolerance=1e-10, relaxation=0.5
              force_tol=1e-6

        Parameters
        ----------
        cfg : dict
            Parsed keyword config with required keys:
            - max_steps   (int)
            - tolerance   (float)
            - relaxation  (float)
            - force_tol   (float)
        """
        missing = [k for k in ("max_steps", "tolerance", "relaxation", "force_tol")
                   if k not in cfg]
        if missing:
            raise ValueError(
                f"*DYNAMIC_RELAXATION requires: {', '.join(missing)}. "
                f"Provide them in the INP file, e.g.: "
                f"max_steps=10000, tolerance=1e-10, relaxation=0.5, force_tol=1e-6")
        self.max_steps = int(cfg["max_steps"])
        self.tol = float(cfg["tolerance"])
        self.beta = float(cfg["relaxation"])
        self.force_tol = float(cfg["force_tol"])

        self.beta = max(0.01, min(1.0, self.beta))

    def run(self, detect_pipeline, assemble_pipeline, u0, K_r, Omega,
            coating_h=None, coating_ep=None, coating_alpha=None):
        """Execute Jacobi-preconditioned DR iteration.

        Parameters
        ----------
        detect_pipeline : callable
            Signature: (u, t, Omega) -> (pen, coords, interp)
        assemble_pipeline : callable
            Signature: (pen, coords, interp, Omega) -> F_total
        u0 : np.ndarray or None
            Initial displacement guess.  If None, starts from zero.
        K_r : np.ndarray
            Reduced stiffness matrix (n_r, n_r).
        Omega : float
            Rotor speed (rad/s).
        coating_h : np.ndarray or None
            Coating thickness grid (DataBus reference).  Pass None
            when no *COATING keyword is used.
        coating_ep : np.ndarray or None
            Coating cumulative plastic strain grid.
        coating_alpha : np.ndarray or None
            Coating back-stress grid.

        Returns
        -------
        tuple[np.ndarray, bool, dict]
            (u_eq, converged, stats_dict)
        """
        n_r = K_r.shape[0]

        # ---- Jacobi preconditioner: D^{-1} = diag(1/K_ii) ----
        K_diag = np.diag(K_r).copy()
        # Guard against zero or negative diagonals (constrained DOFs)
        K_diag = np.where(K_diag > 1e-12, K_diag, 1.0)
        D_inv = 1.0 / K_diag  # (n_r,) — diagonal inverse as vector

        # ---- Initial displacement ----
        u = u0.copy() if u0 is not None else np.zeros(n_r, dtype=np.float64)

        # ---- Save initial coating state (avoid DR wear accumulation) ----
        has_coating = coating_h is not None
        if has_coating:
            h_save = coating_h.copy()
            ep_save = coating_ep.copy()
            alpha_save = (
                coating_alpha.copy() if coating_alpha is not None else None
            )

        def _restore_coating():
            """Restore coating arrays to their pre-DR state.

            Called before each force evaluation to prevent wear
            accumulation during the iterative search.  After convergence,
            wear is applied exactly once.
            """
            if has_coating:
                coating_h[:] = h_save
                coating_ep[:] = ep_save
                if alpha_save is not None:
                    coating_alpha[:] = alpha_save

        # ---- DR iteration ----
        converged = False
        report_interval = max(1, self.max_steps // 10)
        check_interval = max(10, self.max_steps // 200)
        n_fev = 0

        # ── Adaptive beta for stiff contact ──
        # When coating is present, the contact stiffness can exceed
        # the structural stiffness by orders of magnitude, causing
        # the explicit Jacobi iteration to diverge.  An adaptive
        # beta reduction detects divergence and backtracks.
        beta_current = self.beta
        beta_min = max(1e-5, 0.001 * self.beta)
        beta_decay = 0.5
        R_best = np.inf
        u_best = u.copy()
        retreat_count = 0
        max_retreats = 10

        du = np.zeros(n_r)
        R = np.zeros(n_r)
        k = 0
        while k < self.max_steps:
            # Restore coating to initial state — wear is time-dependent,
            # DR is a static solve at t=0; the coating must be intact.
            _restore_coating()

            # ---- Evaluate forces at current displacement ----
            pen, coords, interp = detect_pipeline(u, 0.0, Omega)
            F = assemble_pipeline(pen, coords, interp, Omega)
            n_fev += 1

            if not np.all(np.isfinite(F)):
                print(
                    f"  [DR] WARNING: non-finite force at step {k}; "
                    f"reduce relaxation (currently beta={beta_current})"
                )
                break

            # ---- Residual & Jacobi step ----
            # assemble_pipeline returns -F_contact (CD sign convention).
            # The structural restoring force -K*u was handled by the CD
            # predictor; since we skip it:
            #   R(u) = F_contact_physical - K*u = -assemble_pipeline(u) - K*u
            # Equilibrium: K*u = F_contact_physical  <->  R = 0
            R = -F - K_r @ u
            du = beta_current * D_inv * R
            u = u + du

            # ---- Adaptive beta: divergence detection & backtrack ----
            if k >= check_interval and k % (check_interval // 2) == 0:
                R_inf_check = float(np.linalg.norm(R, np.inf))
                if R_inf_check < R_best:
                    # Improving — record best state
                    R_best = R_inf_check
                    u_best = u.copy()
                elif R_inf_check > R_best * 2.0 and beta_current > beta_min:
                    # Divergence: backtrack to best state, reduce beta.
                    # Skip the Jacobi step this iteration so the next
                    # force eval starts from u_best with the smaller beta.
                    beta_old = beta_current
                    beta_current = max(beta_current * beta_decay, beta_min)
                    u[:] = u_best
                    retreat_count += 1
                    print(
                        f"  [DR] step {k}: |R|={R_inf_check:.3e} > "
                        f"2× best={R_best:.3e}; "
                        f"beta {beta_old:.3f} → {beta_current:.3f}"
                    )
                    if beta_current <= beta_min and retreat_count >= max_retreats:
                        print(
                            f"  [DR] WARNING: adaptive beta exhausted "
                            f"(beta={beta_current:.4f}); try reducing "
                            f"initial relaxation (currently beta={self.beta})"
                        )
                        break
                    k += 1
                    continue  # re-evaluate forces at u_best next iteration

            # ---- Convergence check ----
            if k >= 20 and k % check_interval == 0:
                du_inf = float(np.linalg.norm(du, np.inf))
                R_inf = float(np.linalg.norm(R, np.inf))
                if du_inf < self.tol and R_inf < self.force_tol:
                    converged = True
                    break

            # ---- Progress ----
            if k > 0 and k % report_interval == 0:
                du_inf = float(np.linalg.norm(du, np.inf))
                R_inf = float(np.linalg.norm(R, np.inf))
                print(
                    f"  [DR] step {k}/{self.max_steps} "
                    f"beta={beta_current:.3f} "
                    f"du_inf={du_inf:.3e} "
                    f"|R|={R_inf:.3e} "
                    f"|u|={np.linalg.norm(u):.4e}"
                )

            k += 1

        # ---- Post-convergence: apply wear ONCE from final equilibrium ----
        if has_coating:
            _restore_coating()
            pen, coords, interp = detect_pipeline(u, 0.0, Omega)
            assemble_pipeline(pen, coords, interp, Omega)
            print("  [DR] initial coating wear applied from converged state")

        du_final = float(np.linalg.norm(du, np.inf)) if k > 0 else 0.0
        R_final = float(np.linalg.norm(R, np.inf)) if k > 0 else 0.0
        status = "converged" if converged else "max_steps exhausted"
        print(
            f"  [DR] {status} after {k} steps ({n_fev} force evals), "
            f"du_inf={du_final:.3e}, |R|={R_final:.3e}, "
            f"|u_eq|={np.linalg.norm(u):.4e}"
        )

        stats = {
            "steps": k,
            "n_assemblies": n_fev,
            "du_inf": du_final,
            "|R|_inf": R_final,
            "converged": converged,
            "final_beta": beta_current,
            "retreats": retreat_count,
        }
        return u, converged, stats


# ── Builder ──

def _build_dynamic_relaxation(db, ctx):
    """Build DynamicRelaxation module from context config."""
    mod = DynamicRelaxation(db, ctx)
    dr_cfg = ctx.get("dr_cfg", {}) or {}
    mod.configure(dr_cfg)
    return mod


# ── Register ──

components.register(
    "DYNAMIC_RELAXATION",
    "JACOBI_PRECONDITIONED",
    ModuleSpec(
        builder=_build_dynamic_relaxation,
        protocol="DynamicRelaxationProtocol",
    ),
)

components.register_class("DYNAMIC_RELAXATION", DynamicRelaxation)
