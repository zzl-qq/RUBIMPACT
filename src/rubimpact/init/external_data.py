"""*EXTERNAL_DATA module (framework S4.1)."""
import csv
from pathlib import Path

from rubimpact.infra.databus import DataBus
from rubimpact.infra.sparse_matrix import SparseMatrix
from rubimpact.core.module_base import Module


class ExternalData(Module):
    """Load FE matrices and tip-node coordinates from external files."""

    def configure(self, cfg: dict) -> None:
        """Group submodule entries by slot, then load all data.

        cfg is the INP entry dict with ``sub_list`` (ordered list of
        (slot_name, sub_entry) tuples).  File paths are resolved relative
        to the INP directory stored in ``self.ctx["inp_dir"]``.
        """
        grouped: dict = {}
        for slot, sub in cfg.get("sub_list", []):
            grouped.setdefault(slot, []).append(sub)
        self._load(grouped)

    def _load(self, cfg: dict) -> None:
        inp_dir = self.ctx.get("inp_dir")
        stiffs = {}
        for entry in cfg.get("matrix", []):
            role = entry["ROLE"]
            path = entry["FILE"]
            if inp_dir is not None and not Path(path).is_absolute():
                path = str(Path(inp_dir) / path)
            mat = SparseMatrix.from_mtx_file(path)
            if role == "MASS":
                self.db.set("matrices.mass", mat)
            elif role == "STIFFNESS":
                if "Omega" not in entry:
                    raise ValueError(
                        "*EXTERNAL_DATA matrix entry for STIFFNESS requires "
                        f"'Omega' (rotor speed in rad/s). "
                        f"File: {entry.get('FILE', '?')}")
                omega = float(entry["Omega"])
                stiffs[omega] = mat
        if stiffs:
            self.db.set("matrices.stiffness", stiffs)

        for entry in cfg.get("nodes", []):
            if entry.get("ROLE") == "TIP":
                path = entry["FILE"]
                if inp_dir is not None and not Path(path).is_absolute():
                    path = str(Path(inp_dir) / path)
                tips = self._read_tip_nodes(path)
                self.db.set("nodes.tip", tips)

    @staticmethod
    def _read_tip_nodes(path: str) -> dict:
        tips = {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                nid = int(row[0])
                tips[nid] = (float(row[1]), float(row[2]), float(row[3]))
        return tips


# Register for auto-discovery
from rubimpact.core.registry import components as registry
registry.register_class("EXTERNAL_DATA", ExternalData)
