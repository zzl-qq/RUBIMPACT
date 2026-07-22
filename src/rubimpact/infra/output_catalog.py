"""Standard output variable catalog and dispatcher.

The OutputCatalog defines a framework-level directory of output variable
names (U, CF, PEN, EP, ...) that are independent of which module TYPE
produces them.  Each entry maps a variable name to its category
(HISTORY/FIELD), source phase, and expected dimensionality.

The OutputDispatcher uses the catalog to automatically collect data from
DataBus/StateManager and write to HDF5 at the declared output frequency.
"""

from typing import Any, Optional

import numpy as np


class OutputCatalog:
    """Framework-defined directory of standard output variables."""

    def __init__(self):
        self._entries: dict[str, dict] = {}

    def define(self, name: str, category: str, source: str,
               ndim: Any, description: str = "") -> None:
        """Define a standard output variable.

        Args:
            name: Variable identifier (U, CF, PEN, ENERGY, EP, H, S, ...)
            category: "HISTORY" or "FIELD"
            source: Framework phase name (time_integrator, force_assembler,
                    contact_detector, coating, step_kernel)
            ndim: Dimensionality — int, tuple, or string name (resolved at runtime)
            description: Human-readable description.
        """
        self._entries[name.upper()] = {
            "category": category,
            "source": source,
            "ndim": ndim,
            "description": description,
        }

    def get(self, name: str) -> Optional[dict]:
        """Look up a variable entry.  Returns None if not defined."""
        return self._entries.get(name.upper())

    def list_names(self) -> list[str]:
        """List all defined variable names."""
        return sorted(self._entries.keys())


# Global singleton
output_catalog = OutputCatalog()


# ═══════════════════════════════════════════════════════════════
# Pre-defined standard variables
# ═══════════════════════════════════════════════════════════════

# ── HISTORY ──
output_catalog.define("U",      category="HISTORY", source="time_integrator",
                      ndim="n_tip_dof", description="Tip displacement")
output_catalog.define("CF",     category="HISTORY", source="force_assembler",
                      ndim="n_r", description="Total force vector (ROM space)")
output_catalog.define("PEN",    category="HISTORY", source="contact_detector",
                      ndim="n_tip_nodes", description="Penetration depth")
output_catalog.define("ENERGY", category="HISTORY", source="step_kernel",
                      ndim=4, description="KE, SE, total, dissipation")

# Force decoupling
output_catalog.define("CFN", category="HISTORY", source="force_assembler",
                      ndim="n_r", description="Normal contact force (ROM space)")
output_catalog.define("CFT", category="HISTORY", source="force_assembler",
                      ndim="n_r", description="Friction force (ROM space)")

# ── FIELD ──
output_catalog.define("COATING_H",  category="FIELD", source="coating",
                      ndim="(n_theta,n_x)", description="Coating thickness field")
output_catalog.define("COATING_EP", category="FIELD", source="coating",
                      ndim="(n_theta,n_x)", description="Plastic strain field")
output_catalog.define("COATING_S",  category="FIELD", source="coating",
                      ndim="(n_theta,n_x)", description="Contact stress field")


# ═══════════════════════════════════════════════════════════════
# OutputDispatcher — collects data and writes to HDF5
# ═══════════════════════════════════════════════════════════════


class OutputDispatcher:
    """Collect output data from DataBus/StateManager and write to HDF5.

    The dispatcher knows how to resolve each ``source`` to the actual
    data location (DataBus key or StateManager field).
    """

    def __init__(self):
        self._writer = None
        self._tip_valid = None
        self._tip_idx = None
        self._n_tip_dof = 0
        self._n_tip_nodes = 0
        self._n_r = 0

    def initialize(self, writer, db):
        """Bind HDF5 writer and resolve dimensionalities from DataBus."""
        from rubimpact.modules.contact_detector import tip_dof_indices

        self._writer = writer
        tip_idx = tip_dof_indices(db).ravel()
        self._tip_idx = tip_idx
        self._tip_valid = tip_idx >= 0
        self._n_tip_dof = tip_idx.size
        tip_nodes = db.get("nodes.tip")
        if tip_nodes is None:
            raise ValueError("Missing DataBus key: nodes.tip")
        self._n_tip_nodes = len(tip_nodes)
        n_r = db.get("rom.n_r")
        if n_r is None:
            raise ValueError("Missing DataBus key: rom.n_r")
        self._n_r = int(n_r)

    def _resolve_ndim(self, ndim: Any) -> int | tuple:
        """Resolve string dimension names to actual values."""
        if isinstance(ndim, str):
            mapping = {
                "n_tip_dof": self._n_tip_dof,
                "n_r": self._n_r,
                "n_tip_nodes": self._n_tip_nodes,
            }
            return mapping.get(ndim, 1)
        return ndim

    def _collect(self, source: str, db, state, var_name: str):
        """Collect data for a variable from its source."""
        if source == "time_integrator":
            row = np.zeros(self._n_tip_dof, dtype=np.float64)
            row[self._tip_valid] = state.u_n[self._tip_idx[self._tip_valid]]
            return row
        elif source == "force_assembler":
            mapping = {"CF": "F_total", "CFN": "F_normal", "CFT": "F_friction"}
            return db.get(mapping.get(var_name.upper(), "F_total"))
        elif source == "contact_detector":
            return db.get("penetration")
        elif source == "coating":
            field_map = {
                "COATING_H":  "coating.h",
                "COATING_EP": "coating.ep",
                "COATING_S":  "coating.s",
            }
            return db.get(field_map.get(var_name.upper(),
                                       f"coating.{var_name.lower()}"))
        elif source == "step_kernel":
            return db.get(f"energy.{var_name.lower()}")
        return None

    def open_history(self, var_name: str, ndim: Any):
        """Pre-open an HDF5 history dataset."""
        dim = self._resolve_ndim(ndim)
        self._writer.open_history(var_name.upper(), dim)

    def open_field(self, var_name: str):
        """Prepare for field output."""
        pass  # Field output handled by dump_field

    def write_history(self, var_name: str, db, state):
        """Collect and write one history entry for var_name.

        ENERGY is computed here because it depends on multiple state fields.
        """
        entry = output_catalog.get(var_name)
        if entry is None:
            return
        if var_name.upper() == "ENERGY":
            M_r = db.get("rom.M_r")
            if M_r is None:
                raise ValueError("Missing DataBus key: rom.M_r")
            K_r = db.get("rom.K_r")
            if K_r is None:
                raise ValueError("Missing DataBus key: rom.K_r")
            h = db.get("current_h")
            if h is None:
                raise ValueError("Missing DataBus key: current_h")
            v = (state.u_n - state.u_nm1) / h
            KE = 0.5 * v @ (M_r @ v)
            SE = 0.5 * state.u_n @ (K_r @ state.u_n)
            row = np.array([KE, SE, KE + SE, 0.0], dtype=np.float64)
            self._writer.append_history("ENERGY", row)
        else:
            source = entry["source"]
            data = self._collect(source, db, state, var_name)
            if data is not None:
                self._writer.append_history(var_name.upper(), data)

    def write_field(self, var_name: str, db, step_idx: int):
        """Write a field snapshot for var_name."""
        entry = output_catalog.get(var_name)
        if entry is None:
            return
        data = self._collect(entry["source"], db, None, var_name)
        if data is not None:
            self._writer.dump_field(var_name.upper(), step_idx, data)
