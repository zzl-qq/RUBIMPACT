"""ForceAssembler — protocol-driven orchestration, zero isinstance.

All TYPE differences resolved at configure() time.
assemble() is a single branch-free call for all configurations.
"""
import numpy as np

from rubimpact.infra.databus import DataBus
from rubimpact.core.registry import components
from rubimpact.core.module_base import Module
from rubimpact.core.pipeline_factory import PipelineFactory

# Ensure module and kernel registrations are loaded before resolution.
import rubimpact.modules.contact_force  # noqa: F401 — PCL_CONTACT, PENALTY
import rubimpact.modules.friction_force  # noqa: F401 — COULOMB, STRIBECK
import rubimpact.kernels  # noqa: F401 — constitutive, friction, shared


class ForceAssembler(Module):
    """Orchestrates force computation via protocol-driven pipeline.

    All TYPE differences resolved at configure() time.
    assemble() is a single branch-free call for all configurations.
    """

    def __init__(self, db: DataBus, context: dict | None = None):
        super().__init__(db, context)
        self._contact = None
        self._friction = None
        self._other_modules: list[tuple[str, object]] = []
        self._pipeline = None
        self._dof_idx: np.ndarray | None = None
        self._F_buf: np.ndarray | None = None
        self._F_normal_buf: np.ndarray | None = None
        self._F_friction_buf: np.ndarray | None = None

    # ------------------------------------------------------------------
    def configure(self, cfg: dict) -> None:
        """Resolve submodule TYPEs, build composed JIT pipeline.

        All type-specific logic is captured in the pipeline at build time.
        No isinstance checks, no TYPE dispatch at assemble() time.
        """
        sm = cfg.get("submodules", {})

        # ── Build contact_force module ──
        cf_cfg = sm.get("contact_force")
        if cf_cfg is None:
            raise ValueError(
                "FORCE_ASSEMBLER requires contact_force submodule with TYPE")
        cf_type = cf_cfg.get("TYPE")
        if cf_type is None:
            raise ValueError(
                f"contact_force requires TYPE. "
                f"Registered: {components.list_types('contact_force')}")

        cf_spec = components.resolve_module("contact_force", cf_type)
        if cf_spec is None:
            raise ValueError(
                f"Unknown contact_force TYPE: {cf_type}. "
                f"Registered: {components.list_types('contact_force')}")
        self._contact = cf_spec.builder(self.db, {"contact_cfg": cf_cfg})

        # ── Build friction_force module ──
        ff_cfg = sm.get("friction_force")
        if ff_cfg is None:
            raise ValueError(
                "FORCE_ASSEMBLER requires friction_force submodule with TYPE")
        ff_type = ff_cfg.get("TYPE")
        if ff_type is None:
            raise ValueError(
                "friction_force requires TYPE. "
                f"Registered: {components.list_types('friction_force')}")

        ff_spec = components.resolve_module("friction_force", ff_type)
        if ff_spec is None:
            raise ValueError(
                f"Unknown friction_force TYPE: {ff_type}. "
                f"Registered: {components.list_types('friction_force')}")
        self._friction = ff_spec.builder(self.db, {"friction_cfg": ff_cfg})

        # ── Other force modules (aero, inertial, …) ──
        self._other_modules = []
        for slot_name, slot_cfg in sm.items():
            if slot_name in ("contact_force", "friction_force"):
                continue
            force_type = slot_cfg.get("TYPE")
            if force_type is None:
                raise ValueError(f"Force module '{slot_name}' requires TYPE.")
            spec = components.resolve_module(slot_name, force_type)
            if spec is None:
                raise ValueError(
                    f"Unknown force module: {slot_name} TYPE={force_type}. "
                    f"Registered: {components.list_types(slot_name)}")
            module = spec.builder(self.db, {})
            self._other_modules.append((slot_name, module))

        # ── ROM DOF indices ──
        from rubimpact.modules.contact_detector import tip_dof_indices
        self._dof_idx = tip_dof_indices(self.db)

        # ── Build composed JIT pipeline ──
        # All type differences resolved here — zero runtime dispatch.
        self._pipeline = PipelineFactory.build(
            modules={
                "contact": self._contact,
                "friction": self._friction,
            },
            protocol="ForceAssembler",
            shared_kernels={
                "geometry_decompose": "shared/GEOMETRY_DECOMPOSE",
                "rom_dof_map": "shared/ROM_DOF_MAP",
            },
        )

        # Pre-allocate buffers
        n_r_val = self.db.get("rom.n_r")
        if n_r_val is None:
            raise ValueError(
                "ForceAssembler requires 'rom.n_r' on DataBus. "
                "Run *ROM before *FORCE_ASSEMBLER.")
        n_r = int(n_r_val)
        self._F_buf = np.zeros(n_r, dtype=np.float64)

    # ------------------------------------------------------------------
    def get_pipeline(self):
        """assemble() as pipeline: (pen, coords, interp, Omega) -> F_total.

        Keyword argument ``requested`` is forwarded to :meth:`assemble`.
        When set (e.g. ``{"F_normal", "F_friction"}``), per-component
        force vectors are allocated and written to DataBus for output.
        """
        n_r = int(self.db.get("rom.n_r"))
        def _pipe(pen, coords, interp, Omega, requested=None):
            return self.assemble(
                pen, coords, interp,
                {"n_r": n_r, "Omega": Omega},
                requested=requested)
        return _pipe

    # ------------------------------------------------------------------
    def get_raw_pipeline(self):
        """Return the composed JIT pipeline function (raw, no wrapper)."""
        return self._pipeline

    # ------------------------------------------------------------------
    def assemble(self, pen: np.ndarray, coords: np.ndarray,
                 interp: np.ndarray, context: dict,
                 requested: set | None = None) -> np.ndarray:
        """Sum all force contributions via composed JIT pipeline.

        Single call signature for ALL contact_force/friction_force TYPE
        combinations.  Zero isinstance checks, zero TYPE dispatch.

        Args:
            pen: (n_nodes,) penetration depth from detect().
            coords: (n_nodes, 5) [theta, x, yc, zc, r] from detect().
            interp: (n_nodes, 9|21) coating interpolation from detect().
            context: must contain n_r, Omega.
            requested: DataBus keys needed for output,
                       e.g. {"F_normal", "F_friction"}.

        Returns:
            F_total (n_r,) in ROM space.
        """
        Omega = context["Omega"]
        n_r = context["n_r"]
        need = requested or set()

        # Output buffers
        F_total = self._ensure_buf(self._F_buf, n_r)
        self._F_buf = F_total
        F_normal = (self._ensure_buf(self._F_normal_buf, n_r)
                    if "F_normal" in need else np.zeros(0, dtype=np.float64))
        if "F_normal" in need:
            self._F_normal_buf = F_normal
        F_friction = (self._ensure_buf(self._F_friction_buf, n_r)
                      if "F_friction" in need else np.zeros(0, dtype=np.float64))
        if "F_friction" in need:
            self._F_friction_buf = F_friction

        # ── Single JIT call — zero isinstance, zero dispatch ──
        # Coating arrays: optional — when *COATING is not declared,
        # pass zeros (no coating data, no wear possible).
        ch = self.db.get("coating.h")
        cep = self.db.get("coating.ep")
        calpha = self.db.get("coating.alpha")
        if ch is None:
            ch = np.zeros((1, 1), dtype=np.float64)
        if cep is None:
            cep = np.zeros((1, 1), dtype=np.float64)
        if calpha is None:
            calpha = np.zeros((1, 1), dtype=np.float64)
        self._pipeline(pen, coords, interp, Omega, self._dof_idx,
                       F_total, F_normal, F_friction,
                       ch, cep, calpha)

        # ── Other force modules (Python-side, non-per-contact) ──
        for _slot_name, module in self._other_modules:
            if hasattr(module, 'compute'):
                F_other = module.compute(context)
                F_total += F_other

        # ── Write to DataBus ──
        self.db.set("F_total", F_total)
        if F_normal.size > 0:
            self.db.set("F_normal", F_normal)
        if F_friction.size > 0:
            self.db.set("F_friction", F_friction)

        return F_total

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_buf(buf: np.ndarray | None, n: int) -> np.ndarray:
        """Return a zero-filled buffer of length n, reusing buf if possible."""
        if buf is None or len(buf) != n:
            return np.zeros(n, dtype=np.float64)
        buf.fill(0.0)
        return buf


# Register for auto-discovery
components.register_class("FORCE_ASSEMBLER", ForceAssembler)
