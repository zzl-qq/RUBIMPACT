"""Protocolized TimeIntegrator module — explicit time integration with pipeline protocol.

Replaces the legacy runtime time_integrator.py. Uses components.resolve_kernel()
instead of jit_registry, inherits from Module, and declares a get_pipeline_protocol().
"""
import numpy as np
from rubimpact.infra.databus import DataBus
from rubimpact.core.registry import components, KernelSpec, ModuleSpec
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol


class TimeIntegrator(Module):
    """Explicit time integrator with protocol-driven kernel dispatch.

    Lifecycle:
        1. __init__(db, context) — store DataBus and shared context.
        2. configure(cfg) — parse submodule configs, resolve JIT predictor/corrector.
        3. initialize(h) — pre-factor A_inv, pre-compute coefficient matrices.
        4. predict(u_n, u_nm1) + correct(u_p, F_total) — per-step time marching.
    """

    def __init__(self, db: DataBus, context: dict | None = None):
        super().__init__(db, context)
        self._A_inv = None
        self._M_r = None
        self._K_r = None
        self._D_r = None
        self._h = 0.0
        self._h2 = 0.0

        # JIT function slots
        self._predict_fn = None
        self._correct_fn = None
        self._predictor_type = ""
        self._corrector_type = ""

        # Pre-computed constant coefficient matrices
        self._coeff_n = None
        self._coeff_nm1 = None

        # Pre-allocated output buffers (reused per step)
        self._up_buf = None
        self._un_buf = None

    def configure(self, cfg: dict) -> None:
        """Resolve submodule TYPEs via components registry and store config.

        Replaces the legacy execute(cfg) method.
        """
        sm = cfg.get("submodules", {})

        # Resolve predictor (try pre-computed variant first for perf)
        pred_cfg = sm.get("predictor", {})
        self._predictor_type = pred_cfg.get("TYPE", "")
        if not self._predictor_type:
            raise ValueError(
                "*TIME_INTEGRATOR requires 'predictor' submodule "
                f"with TYPE. Registered: {components.list_types('predictor')}")
        # Prefer pre-computed variant when the base type is LINEAR
        if self._predictor_type == "LINEAR":
            precomp_spec = components.resolve_kernel("predictor", "LINEAR_PRECOMPUTED")
            if precomp_spec is not None:
                self._predict_fn = precomp_spec.fn
                self._predictor_type = "LINEAR_PRECOMPUTED"
            else:
                pred_spec = components.resolve_kernel("predictor", self._predictor_type)
                if pred_spec is None:
                    raise ValueError(
                        f"Unknown predictor TYPE: {self._predictor_type}. "
                        f"Registered: {components.list_types('predictor')}")
                self._predict_fn = pred_spec.fn
        else:
            pred_spec = components.resolve_kernel("predictor", self._predictor_type)
            if pred_spec is None:
                raise ValueError(
                    f"Unknown predictor TYPE: {self._predictor_type}. "
                    f"Registered: {components.list_types('predictor')}")
            self._predict_fn = pred_spec.fn

        # Resolve corrector
        corr_cfg = sm.get("corrector", {})
        self._corrector_type = corr_cfg.get("TYPE", "")
        if not self._corrector_type:
            raise ValueError(
                "*TIME_INTEGRATOR requires 'corrector' submodule "
                f"with TYPE. Registered: {components.list_types('corrector')}")
        corr_spec = components.resolve_kernel("corrector", self._corrector_type)
        if corr_spec is None:
            raise ValueError(
                f"Unknown corrector TYPE: {self._corrector_type}. "
                f"Registered: {components.list_types('corrector')}")
        self._correct_fn = corr_spec.fn

        # Read step size from cfg (may be overridden by initialize())
        self._h = float(cfg.get("_h", cfg.get("h", "1e-6")))

    def initialize(self, h: float) -> None:
        """Pre-factor A_inv = (M_r/h^2 + D_r/(2h))^-1.

        Also pre-computes constant predictor coefficient matrices
        and pre-allocates output buffers for predict/correct.
        """
        self._h = h
        self._h2 = h * h
        self._M_r = self.db.get("rom.M_r")
        self._K_r = self.db.get("rom.K_r")
        self._D_r = self.db.get("rom.D_r")
        A = self._M_r / self._h2 + self._D_r / (2.0 * h)
        self._A_inv = np.linalg.inv(A)

        # Pre-compute constant predictor coefficients
        self._coeff_n = 2.0 * self._M_r / self._h2 - self._K_r
        self._coeff_nm1 = self._D_r / (2.0 * h) - self._M_r / self._h2

        # Pre-allocate output buffers for predict/correct
        n_r = self._M_r.shape[0]
        self._up_buf = np.empty(n_r, dtype=np.float64)
        self._un_buf = np.empty(n_r, dtype=np.float64)

    def predict(self, u_n, u_nm1):
        """Call registered predictor JIT kernel.

        Uses pre-allocated buffer and (when available) pre-computed
        coefficient matrices to avoid per-step allocation and recomputation.
        """
        # Use pre-computed coeffs if kernel variant is LINEAR_PRECOMPUTED
        if self._predictor_type == "LINEAR_PRECOMPUTED":
            self._predict_fn(
                u_n, u_nm1, self._coeff_n, self._coeff_nm1,
                self._A_inv, self._up_buf)
        else:
            self._predict_fn(
                u_n, u_nm1, self._M_r, self._K_r, self._D_r,
                self._h, self._h2, self._A_inv, self._up_buf)
        return self._up_buf

    def correct(self, u_p, F_total):
        """Call registered corrector JIT kernel.

        Uses pre-allocated buffer to avoid per-step allocation.
        """
        self._correct_fn(u_p, F_total, self._A_inv, self._un_buf)
        return self._un_buf

    def get_pipeline_protocol(self) -> PipelineProtocol:
        """Declare the time integration pipeline: predict then correct."""
        return PipelineProtocol(
            stages=[
                PipelineStage(
                    name="predict",
                    kernel_ref=f"predictor/{self._predictor_type}",
                    depends_on=[],
                ),
                PipelineStage(
                    name="correct",
                    kernel_ref=f"corrector/{self._corrector_type}",
                    depends_on=["predict"],
                ),
            ],
            params={"M_r": self._M_r, "K_r": self._K_r, "D_r": self._D_r, "h": self._h},
        )


# ── Builder function ──

def _build_time_integrator(db, ctx):
    """Build a TimeIntegrator module from the shared context."""
    mod = TimeIntegrator(db, ctx)
    mod.configure(ctx.get("time_cfg", {}))
    return mod


    def get_pipeline(self):
        """TimeIntegrator has no pipeline — predict/correct called directly."""
        return None


# ── Register ──

components.register("TIME_INTEGRATOR", "CENTRAL_DIFFERENCE",
    ModuleSpec(builder=_build_time_integrator, protocol="TimeIntegratorProtocol"))

components.register_class("TIME_INTEGRATOR", TimeIntegrator)
