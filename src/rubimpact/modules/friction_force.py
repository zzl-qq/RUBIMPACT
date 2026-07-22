"""Friction force module — COULOMB / STRIBECK."""
import numpy as np
from rubimpact.core.registry import components, ModuleSpec
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol


class FrictionForceModule(Module):
    """Tangential friction force: F_t = mu(v) * F_n.

    Pure function of F_n and v_rel — no DataBus interaction.
    """

    def configure(self, cfg: dict) -> None:
        self.friction_type = cfg.get("TYPE", "COULOMB")
        self.fric_params = self._make_params(cfg, self.friction_type)

    @staticmethod
    def _make_params(cfg: dict, ftype: str) -> np.ndarray:
        if ftype.upper() == "STRIBECK":
            return np.array([
                float(cfg.get("mu_s", "0.3")),
                float(cfg.get("mu_d", "0.25")),
                float(cfg.get("v_s", "1.0")),
            ], dtype=np.float64)
        else:
            return np.array(
                [float(cfg.get("mu", "0.25"))], dtype=np.float64)

    def get_pipeline_protocol(self) -> PipelineProtocol:
        kernel_ref = f"friction/{self.friction_type.upper()}"
        return PipelineProtocol(
            stages=[
                PipelineStage(
                    name="compute_friction_force",
                    kernel_ref=kernel_ref,
                    depends_on=["compute_normal_force"],
                ),
            ],
            params={"fric_params": self.fric_params},
        )


# ── Builders ──

def _build_friction_coulomb(db, ctx):
    mod = FrictionForceModule(db, ctx)
    ff_cfg = ctx.get("friction_cfg", {})
    mod.configure({
        "TYPE": "COULOMB",
        "mu": ff_cfg.get("mu", "0.25"),
    })
    return mod


def _build_friction_stribeck(db, ctx):
    mod = FrictionForceModule(db, ctx)
    ff_cfg = ctx.get("friction_cfg", {})
    mod.configure({
        "TYPE": "STRIBECK",
        "mu_s": ff_cfg.get("mu_s", "0.3"),
        "mu_d": ff_cfg.get("mu_d", "0.25"),
        "v_s": ff_cfg.get("v_s", "1.0"),
    })
    return mod


components.register("friction_force", "COULOMB",
    ModuleSpec(builder=_build_friction_coulomb, protocol="FrictionProtocol"))
components.register("friction_force", "STRIBECK",
    ModuleSpec(builder=_build_friction_stribeck, protocol="FrictionProtocol"))
