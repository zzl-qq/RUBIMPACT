"""*COATING module (framework S4.3)."""
import numpy as np
from rubimpact.infra.databus import DataBus
from rubimpact.core.module_base import Module
from rubimpact.core.registry import components


class Coating(Module):
    """Initialise coating geometry (grid, thickness, plastic strain, back-stress)."""

    def configure(self, cfg: dict) -> None:
        ctype = cfg.get("TYPE")
        if ctype is None:
            raise ValueError("*COATING requires TYPE parameter. "
                             f"Registered: {components.list_types('COATING')}")
        builder = components.get("COATING", ctype)
        if builder is None:
            raise ValueError(
                f"Unknown COATING TYPE: {ctype}. "
                f"Registered: {components.list_types('COATING')}")
        cfg.update(builder(cfg))

        h_coat = float(cfg["h_coat"])
        L = float(cfg["L"])
        n_theta = int(cfg["n_theta"])
        n_x = int(cfg["n_x"])
        dtheta = 2.0 * np.pi / n_theta
        dx = L / n_x
        casing = self.db.get("casing.geometry", {})
        if "R0" not in casing:
            raise ValueError(
                "*COATING requires casing R0.  Run *CASING before *COATING "
                "to set 'casing.geometry' on DataBus.")
        R0 = float(casing["R0"])
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        x = np.linspace(0.0, L, n_x, endpoint=False)
        grid = {"h_coat": h_coat, "L": L, "n_theta": n_theta, "n_x": n_x,
                "dtheta": dtheta, "dx": dx, "A_cell": R0 * dtheta * dx, "R0": R0,
                "theta": theta, "x": x}
        self.db.set("coating.grid", grid)
        self.db.set("coating.h", np.full((n_theta, n_x), h_coat, dtype=np.float64))
        self.db.set("coating.ep", np.zeros((n_theta, n_x), dtype=np.float64))
        self.db.set("coating.alpha", np.zeros((n_theta, n_x), dtype=np.float64))


# ── COATING TYPE builder ──

def _build_uniform_grid(cfg):
    """UNIFORM_GRID — validate required params (all in INP cfg)."""
    float(cfg["h_coat"]); float(cfg["L"])
    int(cfg["n_theta"]); int(cfg["n_x"])
    return {}  # no extra params needed

components.register("COATING", "UNIFORM_GRID", _build_uniform_grid)
components.register_class("COATING", Coating)
