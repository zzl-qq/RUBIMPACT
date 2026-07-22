"""*CASING module — axial + circumferential shape composition.

R(x, θ) = R₀ + f_axial(x) + f_circ(θ)

f_axial and f_circ are resolved via ComponentRegistry with categories
``axial_shape`` and ``circumferential_shape`` respectively.  The
ContactDetector reads the composed ``get_radius`` and ``R_grid`` from
DataBus — it has zero knowledge of the decomposition.
"""
import numpy as np
from rubimpact.infra.databus import DataBus
from rubimpact.core.module_base import Module
from rubimpact.core.registry import components


class Casing(Module):
    """Casing geometry — composes axial + circumferential shape functions."""

    def configure(self, cfg: dict) -> None:
        R0 = float(cfg["R0"])  # required — KeyError if missing
        sm = cfg.get("submodules", {})

        # ── Axial shape (required) ──
        ax_cfg = sm.get("axial_shape")
        if ax_cfg is None:
            raise ValueError(
                "*CASING requires 'axial_shape' submodule with TYPE. "
                f"Registered: {components.list_types('axial_shape')}")
        ax_type = ax_cfg.get("TYPE")
        if ax_type is None:
            raise ValueError(
                "axial_shape requires TYPE parameter. "
                f"Registered: {components.list_types('axial_shape')}")
        ax_builder = components.get("axial_shape", ax_type)
        if ax_builder is None:
            raise ValueError(
                f"Unknown axial_shape TYPE: {ax_type}. "
                f"Registered: {components.list_types('axial_shape')}")
        axial = ax_builder(ax_cfg)

        # ── Circumferential shape (optional, default UNIFORM) ──
        circ_cfg = sm.get("circumferential_shape", {})
        circ_type = circ_cfg.get("TYPE", "UNIFORM")
        circ_builder = components.get("circumferential_shape", circ_type)
        if circ_builder is None:
            raise ValueError(
                f"Unknown circumferential_shape TYPE: {circ_type}. "
                f"Registered: {components.list_types('circumferential_shape')}")
        circ = circ_builder(circ_cfg)

        # ── Combined geometry ──
        def _get_radius(x, theta):
            return R0 + axial["get_delta"](x) + circ["get_delta"](theta)

        def _R_grid(theta, x):
            ax_delta = np.array([axial["get_delta"](float(xi)) for xi in x],
                                dtype=np.float64)
            circ_delta = np.array([circ["get_delta"](float(ti)) for ti in theta],
                                  dtype=np.float64)
            return (R0
                    + np.tile(ax_delta, (len(theta), 1))
                    + np.tile(circ_delta[:, None], (1, len(x))))

        self.db.set("casing.geometry", {
            "type": f"{axial['type']}+{circ['type']}",
            "R0": R0,
            "get_radius": _get_radius,
            "R_grid": _R_grid,
        })

    @staticmethod
    def build_R_grid(db: DataBus) -> None:
        """Pre-compute casing radius on the coating grid.

        Called after *COATING initialisation so that the grid dimensions
        are known.  Stores a ``casing.R_grid`` ndarray of shape
        (n_theta, n_x) and ``casing.R_grid_params`` for use by the JIT
        bilinear interpolator kernel.
        """
        geom = db.get("casing.geometry")
        grid = db.get("coating.grid")
        if geom is None or grid is None:
            return
        theta = grid["theta"]
        x = grid["x"]
        n_theta = len(theta)
        n_x = len(x)
        dtheta = float(grid["dtheta"])
        dx = float(grid["dx"])
        x_min = float(x[0])
        theta_min = float(theta[0])

        R_grid_fn = geom.get("R_grid")
        if R_grid_fn is not None:
            R = R_grid_fn(theta, x)
        else:
            get_r = geom["get_radius"]
            R = np.empty((n_theta, n_x), dtype=np.float64)
            for i in range(n_theta):
                th = theta[i]
                for j in range(n_x):
                    R[i, j] = get_r(x[j], th)
        db.set("casing.R_grid", R)

        db.set("casing.R_grid_params", np.array(
            [dtheta, dx, x_min, theta_min, float(n_theta), float(n_x)],
            dtype=np.float64))


# ── Casing shape builders ──

def _build_cylindrical(cfg):
    return {"type": "CYLINDRICAL", "get_delta": lambda x: 0.0}

def _build_taper(cfg):
    slope = float(cfg["slope"])
    return {"type": "AXIAL_TAPER", "get_delta": lambda x, s=slope: s * x}

def _build_uniform(cfg):
    return {"type": "UNIFORM", "get_delta": lambda theta: 0.0}

def _build_lobe(cfg):
    N_lobe = int(cfg["N_lobe"])
    d0 = float(cfg["d0"])
    return {"type": "LOBE",
            "get_delta": lambda theta, n=N_lobe, d=d0: d * np.sin(n * theta)}

components.register("axial_shape", "CYLINDRICAL", _build_cylindrical)
components.register("axial_shape", "AXIAL_TAPER", _build_taper)
components.register("circumferential_shape", "UNIFORM", _build_uniform)
components.register("circumferential_shape", "LOBE", _build_lobe)

# Register class for auto-discovery
components.register_class("CASING", Casing)
