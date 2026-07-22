"""*CONSTITUTIVE module — Plastic Coating Law.

The JIT kernel for PCL return mapping is in kernels/constitutive.py.
This module handles TYPE resolution, submodule dispatch, and parameter
aggregation, storing resolved params on DataBus for downstream use.
"""
from rubimpact.infra.databus import DataBus
from rubimpact.core.module_base import Module
from rubimpact.core.registry import components


class ConstitutiveModule(Module):
    """Top-level CONSTITUTIVE dispatcher.

    Resolves the constitutive model TYPE and its submodules via the
    module registry, returning material parameters for downstream
    ForceAssembler and JIT kernel initialization.
    """

    def __init__(self, db: DataBus, context: dict | None = None):
        super().__init__(db, context)
        self._pcl_fn = None

    def configure(self, cfg: dict) -> None:
        """Resolve constitutive model and submodules, store on DataBus.

        Uses category ``const_model`` (not ``CONSTITUTIVE``) for the
        module TYPE builder to avoid colliding with the JIT kernel
        category ``constitutive``.
        """
        const_type = cfg.get("TYPE")
        if const_type is None:
            raise ValueError(
                "*CONSTITUTIVE requires TYPE parameter. "
                f"Registered: {components.list_types('const_model')}")
        const_builder = components.get("const_model", const_type)
        if const_builder is None:
            raise ValueError(
                f"Unknown CONSTITUTIVE TYPE: {const_type}. "
                f"Registered: {components.list_types('const_model')}")

        const_resolved = const_builder(cfg)
        E = float(const_resolved["E"])
        Y = float(const_resolved["Y"])

        # Resolve PCL JIT function
        self._pcl_fn = components.get("constitutive", const_type)
        if self._pcl_fn is None:
            raise ValueError(
                f"No JIT kernel for CONSTITUTIVE TYPE: {const_type}. "
                f"Registered: {components.list_types('constitutive')}")

        # Hardening submodule
        sm = cfg.get("submodules", {})
        hardening = sm.get("hardening", {})
        hw_type = hardening.get("TYPE")
        if hw_type is None:
            raise ValueError(
                "*CONSTITUTIVE requires 'hardening' submodule with TYPE. "
                f"Registered: {components.list_types('hardening')}")
        hw_builder = components.get("hardening", hw_type)
        if hw_builder is None:
            raise ValueError(
                f"Unknown hardening TYPE: {hw_type}. "
                f"Registered: {components.list_types('hardening')}")
        hw_params = hw_builder(hardening)
        K_plas = float(hw_params["K_plas"])

        # Submodule resolution
        resolved = {"E": E, "Y": Y, "K_plas": K_plas}
        for slot, cat_name in (("wear_law", "wear_law"),
                               ("wear_distributor", "wear_distributor"),
                               ("state_updater", "state_updater")):
            sm_cfg = sm.get(slot, {})
            sm_type = sm_cfg.get("TYPE")
            if sm_type is None:
                raise ValueError(
                    f"*CONSTITUTIVE requires '{slot}' submodule "
                    f"with TYPE. "
                    f"Registered: {components.list_types(cat_name)}")
            builder = components.get(cat_name, sm_type)
            if builder is None:
                raise ValueError(
                    f"Unknown {slot} TYPE: {sm_type}. "
                    f"Registered: {components.list_types(cat_name)}")
            resolved[slot] = builder(sm_cfg)

        self.db.set("const_params", resolved)

    def kernel_info(self) -> dict:
        """Return PCL function and material params for downstream."""
        return {"pcl_fn": self._pcl_fn}


# ── CONSTITUTIVE sub-builders ──

def _build_plastic_coating_law(cfg):
    return {"E": float(cfg["E"]), "Y": float(cfg["Y"])}

def _build_linear_isotropic(cfg):
    return {"K_plas": float(cfg["K_plas"])}

def _build_plastic_strain_wear(cfg):
    return {"TYPE": "PLASTIC_STRAIN"}

def _build_wear_none(cfg):
    return {"TYPE": "NONE"}

def _build_bilinear_weight(cfg):
    return {"TYPE": "BILINEAR_WEIGHT"}

def _build_plastic_strain_ratio(cfg):
    return {"TYPE": "PLASTIC_STRAIN_RATIO"}

components.register("const_model", "PLASTIC_COATING_LAW",
                    _build_plastic_coating_law)
components.register("hardening", "LINEAR_ISOTROPIC", _build_linear_isotropic)
components.register("wear_law", "PLASTIC_STRAIN", _build_plastic_strain_wear)
components.register("wear_law", "NONE", _build_wear_none)
components.register("wear_distributor", "BILINEAR_WEIGHT",
                    _build_bilinear_weight)
components.register("state_updater", "PLASTIC_STRAIN_RATIO",
                    _build_plastic_strain_ratio)
components.register_class("CONSTITUTIVE", ConstitutiveModule)
