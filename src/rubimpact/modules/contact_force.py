"""Contact force modules — PCL_CONTACT and PENALTY."""
import numpy as np
from rubimpact.infra.databus import DataBus
from rubimpact.core.registry import components, ModuleSpec
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol
from rubimpact.core.param_utils import required_scalar


class ContactForceModule(Module):
    """PCL-based normal contact force: F_n = A_cell * sigma.

    Requires coating (h, ep, alpha grids).  Wear is optional.
    """

    def configure(self, cfg: dict) -> None:
        # Read material parameters — DataBus (const_params) first,
        # then direct INP cfg.  No silent defaults.
        const_params = self.db.get("const_params", {}) or {}

        self.E = required_scalar(
            self.db, cfg, "E", db_key="const_params",
            source="*CONSTITUTIVE keyword in INP file")
        self.Y = required_scalar(
            self.db, cfg, "Y", db_key="const_params",
            source="*CONSTITUTIVE keyword in INP file")
        self.K_plas = required_scalar(
            self.db, cfg, "K_plas", db_key="const_params",
            source="*CONSTITUTIVE keyword in INP file")

        # Coating parameters — required when coating grid exists
        grid = self.db.get("coating.grid")
        if grid is None:
            raise ValueError(
                "ContactForceModule (PCL_CONTACT) requires coating.grid "
                "on DataBus.  Run *COATING before contact force modules.")
        self.A_cell = float(grid["A_cell"])
        self.h_coat = float(grid["h_coat"])
        self.coating_h = self.db.get("coating.h")
        self.coating_ep = self.db.get("coating.ep")
        self.coating_alpha = self.db.get("coating.alpha")

        self.pcl_params = np.array(
            [self.E, self.Y, self.K_plas, self.h_coat], dtype=np.float64)

        # Wear config — from CONSTITUTIVE (DataBus), INP cfg, or default NONE.
        # wear_law is a sub-module enablement, not a physical parameter.
        wear_cfg = (const_params.get("wear_law")
                    or cfg.get("wear_law")
                    or {"TYPE": "NONE"})
        self.wear_enabled = wear_cfg.get("TYPE", "NONE") != "NONE"

        if self.coating_ep is not None:
            n_th = self.coating_ep.shape[0]
        else:
            n_th = 1
        self.wear_params = np.array([float(n_th), self.h_coat], dtype=np.float64)

        # Resolve wear kernel — use unified kernel for both bilinear/bicubic
        self.interpolator_type = self.db.get("coating.interpolator_type")
        if self.interpolator_type is None:
            raise ValueError(
                "ContactForceModule requires 'coating.interpolator_type' "
                "on DataBus.  Set by *CONTACT_DETECTOR.")
        wear_base = wear_cfg.get("TYPE", "NONE")
        if wear_base != "NONE":
            # Always use UNIFIED kernel — handles both interpolator types
            # with a single call signature, avoiding Numba branch mismatch
            self.wear_kernel = components.resolve_kernel("wear", "UNIFIED")
            if self.wear_kernel is None:
                raise ValueError("No JIT kernel for wear: UNIFIED")
        else:
            self.wear_kernel = None

    def get_pipeline_protocol(self) -> PipelineProtocol:
        stages = [
            PipelineStage(
                name="compute_normal_force",
                kernel_ref="constitutive/PLASTIC_COATING_LAW",
                depends_on=[],
            ),
        ]
        if self.wear_enabled and self.wear_kernel is not None:
            stages.append(PipelineStage(
                name="apply_wear",
                kernel_ref="wear/UNIFIED",
                depends_on=["compute_normal_force"],
                optional=True,
            ))
        return PipelineProtocol(
            stages=stages,
            params={
                "pcl_params": self.pcl_params,
                "A_cell": self.A_cell,
                "wear_enabled": self.wear_enabled,
                "wear_params": self.wear_params,
                "coating_h": self.coating_h,
                "coating_ep": self.coating_ep,
                "coating_alpha": self.coating_alpha,
            },
        )


class PenaltyContactForceModule(Module):
    """Penalty-based normal contact force: F_n = k * delta."""

    def configure(self, cfg: dict) -> None:
        self.k_penalty = float(cfg["k_penalty"])

    def get_pipeline_protocol(self) -> PipelineProtocol:
        # PENALTY has no external kernel — F_n = k * delta is inline in PipelineFactory
        return PipelineProtocol(
            stages=[
                PipelineStage(
                    name="compute_normal_force",
                    kernel_ref=None,  # inline logic
                    depends_on=[],
                ),
            ],
            params={"k_penalty": self.k_penalty},
        )


# ── Builder functions ──

def _build_contact_force_pcl(db, ctx):
    module = ContactForceModule(db, ctx)
    module.configure(ctx.get("contact_cfg", {}))
    return module


def _build_contact_force_penalty(db, ctx):
    module = PenaltyContactForceModule(db, ctx)
    module.configure(ctx.get("contact_cfg", {}))
    return module


# ── Register ──

components.register("contact_force", "PCL_CONTACT",
    ModuleSpec(builder=_build_contact_force_pcl, protocol="NormalForceProtocol"))
components.register("contact_force", "PENALTY",
    ModuleSpec(builder=_build_contact_force_penalty, protocol="NormalForceProtocol"))
