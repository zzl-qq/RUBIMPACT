"""*MATRIX_ASSEMBLY module (framework S4.4)."""
import numpy as np
from scipy.sparse import csr_matrix
from rubimpact.infra.databus import DataBus
from rubimpact.infra.sparse_matrix import SparseMatrix
from rubimpact.core.module_base import Module
from rubimpact.core.registry import components


class MatrixAssembly(Module):
    """Assemble and store system matrices (M, K, D) on DataBus."""

    def configure(self, cfg: dict) -> None:
        Omega = self.ctx.get("Omega", 0.0)
        sm = cfg.get("submodules", {})

        # --- Stiffness (required) ---
        stiff_cfg = sm.get("stiffness", {})
        stiff_type = stiff_cfg.get("TYPE")
        if stiff_type is None:
            raise ValueError(
                "*MATRIX_ASSEMBLY requires 'stiffness' submodule with TYPE. "
                f"Registered: {components.list_types('stiffness')}")
        if components.get("stiffness", stiff_type) is None:
            raise ValueError(
                f"Unknown stiffness TYPE: {stiff_type}. "
                f"Registered: {components.list_types('stiffness')}")
        stiffnesses = self.db.get("matrices.stiffness", {})
        if stiff_type == "CENTRIFUGAL_POLY":
            K = self._build_centrifugal_poly(stiffnesses, Omega)
        else:
            K = list(stiffnesses.values())[0]
        self.db.set("matrices.K_omega", K)

        # --- Mass (required — validate TYPE for consistency) ---
        mass_cfg = sm.get("mass", {})
        mass_type = mass_cfg.get("TYPE")
        if mass_type is None:
            raise ValueError(
                "*MATRIX_ASSEMBLY requires 'mass' submodule with TYPE. "
                f"Registered: {components.list_types('mass')}")
        if components.get("mass", mass_type) is None:
            raise ValueError(
                f"Unknown mass TYPE: {mass_type}. "
                f"Registered: {components.list_types('mass')}")

        # --- Damping (required) ---
        damp_cfg = sm.get("damping", {})
        damp_type = damp_cfg.get("TYPE")
        if damp_type is None:
            raise ValueError(
                "*MATRIX_ASSEMBLY requires 'damping' submodule with TYPE. "
                f"Registered: {components.list_types('damping')}")
        builder = components.get("damping", damp_type)
        if builder is None:
            raise ValueError(
                f"Unknown damping TYPE: {damp_type}. "
                f"Registered: {components.list_types('damping')}")
        damp = builder(damp_cfg)

        if damp["TYPE"] == "RAYLEIGH":
            alpha = float(damp["alpha"])
            beta = float(damp["beta"])
            M = self.db.get("matrices.mass")
            D_csr = alpha * M.csr + beta * K.csr
            self.db.set("matrices.D_full", SparseMatrix(D_csr))
        else:
            self.db.set("matrices.D_full",
                         SparseMatrix(csr_matrix((K.shape[0], K.shape[0]),
                                                dtype=np.float64)))

    @staticmethod
    def _build_centrifugal_poly(stiffs, Omega):
        omegas = sorted(stiffs.keys())
        if len(omegas) == 1:
            return stiffs[omegas[0]]
        K0 = stiffs[omegas[0]]
        om = omegas[-1]
        if len(omegas) >= 3:
            K_half = stiffs[omegas[1]]
            K_max = stiffs[om]
            K1 = (16.0 * K_half.csr - K_max.csr - 15.0 * K0.csr) / (3.0 * om**2)
            K2 = 4.0 * (K_max.csr - 4.0 * K_half.csr + 3.0 * K0.csr) / (3.0 * om**4)
            return SparseMatrix((K0.csr + Omega**2 * K1 + Omega**4 * K2).tocsr())
        K_max = stiffs[om]
        K1 = (K_max.csr - K0.csr) / (om**2)
        return SparseMatrix((K0.csr + Omega**2 * K1).tocsr())


# ── Matrix assembly TYPE builders ──

def _build_stiff_direct(cfg): return {"TYPE": "DIRECT"}
def _build_stiff_centrifugal(cfg): return {"TYPE": "CENTRIFUGAL_POLY"}
def _build_mass_original(cfg): return {"TYPE": "ORIGINAL"}
def _build_damp_rayleigh(cfg):
    return {"TYPE": "RAYLEIGH",
            "alpha": float(cfg["alpha"]), "beta": float(cfg["beta"])}
def _build_damp_none(cfg): return {"TYPE": "NONE"}

components.register("stiffness", "DIRECT", _build_stiff_direct)
components.register("stiffness", "CENTRIFUGAL_POLY", _build_stiff_centrifugal)
components.register("mass", "ORIGINAL", _build_mass_original)
components.register("damping", "RAYLEIGH", _build_damp_rayleigh)
components.register("damping", "NONE", _build_damp_none)
components.register_class("MATRIX_ASSEMBLY", MatrixAssembly)
