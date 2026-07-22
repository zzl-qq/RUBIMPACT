"""CONTACT_DETECTOR module — protocolized via ComponentRegistry.

Per time step, detects contact between blade tip nodes and casing.

The three submodule slots (kinematics, interpolator, gap_function) now
resolve their TYPE to actual @njit functions from ComponentRegistry at
configure time.  Swapping any submodule TYPE only requires registering a
new Numba function + YAML candidate entry.
"""

import numpy as np

from rubimpact.infra.databus import DataBus
from rubimpact.core.registry import components
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol

DOF_PER_NODE = 3


def tip_dof_indices(db: DataBus) -> np.ndarray:
    """Index of each tip-node DOF inside u_p / F_c_ROM vectors."""
    tip_nodes = db.get("nodes.tip", {})
    tip_dof_map = db.get("rom.tip_dof_map", []) or []
    rom_enabled = bool(db.get("rom.enabled", True))
    pos = {int(d): k for k, d in enumerate(tip_dof_map)}
    idx = np.full((len(tip_nodes), DOF_PER_NODE), -1, dtype=np.int64)
    for i, nid in enumerate(sorted(tip_nodes.keys())):
        for d in range(DOF_PER_NODE):
            g = (nid - 1) * DOF_PER_NODE + d
            if g in pos:
                idx[i, d] = pos[g] if rom_enabled else g
    return idx


class ContactDetector(Module):
    """Per-step penetration detection with real submodule dispatch.

    The three submodule steps (kinematics -> interpolator -> gap_function)
    each resolve to a registered @njit function at configure time.  Their
    output arrays are pre-allocated once and reused per step.

    Implements the Module protocol:
        - configure(cfg): resolve kernels from ComponentRegistry
        - get_pipeline_protocol(): declare 4-stage pipeline
        - detect(): unchanged public API for backwards compatibility
    """

    def __init__(self, db: DataBus, context=None):
        super().__init__(db, context)
        self._db = db  # backward-compat alias (used by detect(), etc.)

        # Geometry (set in initialize)
        self._theta0 = np.zeros(0)
        self._x0 = np.zeros(0)
        self._y0 = np.zeros(0)
        self._z0 = np.zeros(0)
        self._dof_idx = np.zeros((0, DOF_PER_NODE), dtype=np.int64)
        self._has_coating = False
        self._grid = None

        # Casing radius getter (Python fallback)
        self._get_radius = lambda x, theta: 0.0

        # JIT casing radius interpolator (fast path)
        self._radius_kernel = None
        self._R_grid = None
        self._R_grid_params = None

        # JIT function slots (resolved at configure time)
        self._kinematics_fn = None
        self._interpolate_fn = None
        self._gap_fn = None
        self._kinematics_type = ""
        self._interpolator_type = ""
        self._gap_type = ""

        # Pre-allocated buffers
        self._n_nodes = 0
        self._coords_buf: np.ndarray | None = None         # (n_nodes, 5)
        self._interp_buf: np.ndarray | None = None          # (n_nodes, 9) or (n_nodes, 21)
        self._pen_buf: np.ndarray | None = None             # (n_nodes,)
        self._R_buf: np.ndarray | None = None               # (n_nodes,)
        self._grid_params: np.ndarray | None = None          # (6,)

    def configure(self, cfg: dict) -> None:
        """Resolve submodule TYPEs via ComponentRegistry and initialize."""
        sm = cfg.get("submodules", {})

        # Resolve kinematics
        kin_cfg = sm.get("kinematics", {})
        self._kinematics_type = kin_cfg.get("TYPE", "")
        if not self._kinematics_type:
            raise ValueError(
                "*CONTACT_DETECTOR requires 'kinematics' submodule "
                f"with TYPE. Registered: {components.list_types('kinematics')}")
        kin_spec = components.resolve_kernel("kinematics", self._kinematics_type)
        if kin_spec is None:
            raise ValueError(
                f"Unknown kinematics TYPE: {self._kinematics_type}. "
                f"Registered: {components.list_types('kinematics')}")
        self._kinematics_fn = kin_spec.fn

        # Resolve interpolator
        int_cfg = sm.get("interpolator", {})
        self._interpolator_type = int_cfg.get("TYPE", "")
        if not self._interpolator_type:
            raise ValueError(
                "*CONTACT_DETECTOR requires 'interpolator' submodule "
                f"with TYPE. Registered: {components.list_types('interpolator')}")
        int_spec = components.resolve_kernel("interpolator", self._interpolator_type)
        if int_spec is None:
            raise ValueError(
                f"Unknown interpolator TYPE: {self._interpolator_type}. "
                f"Registered: {components.list_types('interpolator')}")
        self._interpolate_fn = int_spec.fn

        # Resolve gap function
        gap_cfg = sm.get("gap_function", {})
        self._gap_type = gap_cfg.get("TYPE", "")
        if not self._gap_type:
            raise ValueError(
                "*CONTACT_DETECTOR requires 'gap_function' submodule "
                f"with TYPE. Registered: {components.list_types('gap_function')}")
        gap_spec = components.resolve_kernel("gap_function", self._gap_type)
        if gap_spec is None:
            raise ValueError(
                f"Unknown gap_function TYPE: {self._gap_type}. "
                f"Registered: {components.list_types('gap_function')}")
        self._gap_fn = gap_spec.fn

        self.initialize()

    def get_pipeline_protocol(self) -> PipelineProtocol:
        """Declare the 4-stage contact detection pipeline."""
        return PipelineProtocol(
            stages=[
                PipelineStage(name="compute_kinematics",
                    kernel_ref=f"kinematics/{self._kinematics_type}", depends_on=[]),
                PipelineStage(name="compute_casing_radius",
                    kernel_ref="casing_radius/RGRID_BILINEAR", depends_on=["compute_kinematics"]),
                PipelineStage(name="interpolate_coating",
                    kernel_ref=f"interpolator/{self._interpolator_type}", depends_on=["compute_kinematics"]),
                PipelineStage(name="compute_gap",
                    kernel_ref=f"gap_function/{self._gap_type}", depends_on=["compute_casing_radius", "interpolate_coating"]),
            ],
            params={"theta0": self._theta0, "x0": self._x0, "y0": self._y0, "z0": self._z0,
                    "dof_idx": self._dof_idx, "grid_params": self._grid_params,
                    "R_grid": self._R_grid, "R_grid_params": self._R_grid_params},
        )

    def initialize(self) -> None:
        """Read tip node geometry from DataBus, allocate buffers."""
        tip_nodes = self._db.get("nodes.tip", {})
        nids = sorted(tip_nodes.keys())
        self._n_nodes = len(nids)
        self._theta0 = np.array(
            [np.arctan2(tip_nodes[n][2], tip_nodes[n][1]) for n in nids],
            dtype=np.float64)
        self._x0 = np.array([tip_nodes[n][0] for n in nids], dtype=np.float64)
        self._y0 = np.array([tip_nodes[n][1] for n in nids], dtype=np.float64)
        self._z0 = np.array([tip_nodes[n][2] for n in nids], dtype=np.float64)
        self._dof_idx = tip_dof_indices(self._db)
        self._grid = self._db.get("coating.grid")
        self._has_coating = self._grid is not None

        # Store interpolator type so downstream modules can select wear/stress kernel
        if self._has_coating:
            self._db.set("coating.interpolator_type", self._interpolator_type)

        geom = self._db.get("casing.geometry", {})
        _get_radius = geom.get("get_radius")
        if _get_radius is None:
            raise ValueError(
                "ContactDetector requires 'get_radius' in casing.geometry. "
                "Run *CASING before *CONTACT_DETECTOR.")
        self._get_radius = _get_radius

        # ── JIT casing radius interpolator (fast path) ──
        radius_spec = components.resolve_kernel("casing_radius", "RGRID_BILINEAR")
        self._radius_kernel = radius_spec.fn if radius_spec is not None else None
        self._R_grid = self._db.get("casing.R_grid")
        self._R_grid_params = self._db.get("casing.R_grid_params")

        # Pre-allocate buffers
        self._coords_buf = np.zeros((self._n_nodes, 5), dtype=np.float64)
        if self._interpolator_type == "BICUBIC_BSPLINE":
            self._interp_buf = np.zeros((self._n_nodes, 21), dtype=np.float64)
        else:
            self._interp_buf = np.zeros((self._n_nodes, 9), dtype=np.float64)
        self._pen_buf = np.zeros(self._n_nodes, dtype=np.float64)
        self._R_buf = np.empty(self._n_nodes, dtype=np.float64)

        if self._has_coating and self._grid is not None:
            coating_h = self._db.get("coating.h")
            n_theta, n_x = coating_h.shape

            if self._interpolator_type == "BICUBIC_BSPLINE":
                if n_theta < 4 or n_x < 4:
                    raise ValueError(
                        f"BICUBIC_BSPLINE interpolator requires n_theta >= 4 and "
                        f"n_x >= 4, but got n_theta={n_theta}, n_x={n_x}. "
                        f"Use TYPE=BILINEAR for smaller grids.")

            self._grid_params = np.array([
                float(self._grid["dtheta"]),
                float(self._grid["dx"]),
                float(self._grid["x"][0]),
                float(n_theta),
                float(n_x),
                1.0,  # has_coating
            ], dtype=np.float64)
        else:
            self._grid_params = np.array([1.0] * 5 + [0.0], dtype=np.float64)

        self._check_initial_interference(nids)

    def _check_initial_interference(self, nids):
        """Warn if any tip node is initially in interference at t=0."""
        h_coat = self._grid["h_coat"] if self._grid else 0.0
        max_pen = 0.0
        for i, nid in enumerate(nids):
            x, y, z = self._x0[i], self._y0[i], self._z0[i]
            r_val = np.hypot(y, z)
            theta = self._theta0[i] % (2.0 * np.pi)
            R = float(self._get_radius(float(x), float(theta)))
            gap = R - h_coat - r_val
            if gap < 0.0:
                pen = -gap
                max_pen = max(max_pen, pen)
                if pen > 0.5 * h_coat:
                    print(f"  [WARN] node {nid}: initial penetration {pen:.1e} mm "
                          f"> 50% of coating thickness {h_coat} mm")
        if max_pen > 0.0:
            print(f"  [init] max static interference = {max_pen:.3e} mm "
                  f"(coating nominal = {h_coat} mm)")
        else:
            print(f"  [init] all tip nodes clear at t=0 "
                  f"(coating nominal = {h_coat} mm)")

    def _compute_R_tip(self, u_p, t, Omega) -> np.ndarray:
        """Pre-compute casing radius at each tip node (Python, not JIT).

        Uses casing.geometry.get_radius so any registered casing type
        works --- zero casing knowledge in the JIT hot loop.
        """
        R = self._R_buf
        n_r = len(u_p)
        for i in range(self._n_nodes):
            kx = self._dof_idx[i, 0]
            ux = u_p[kx] if 0 <= kx < n_r else 0.0
            x = self._x0[i] + ux
            theta = Omega * t + self._theta0[i]
            R[i] = self._get_radius(float(x), float(theta % (2.0 * np.pi)))
        return R

    def detect(self, u_p, t, Omega,
               copy_to_bus: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run three JIT steps: kinematics -> interpolate -> gap.

        Args:
            u_p: Predicted displacement in ROM space (n_r,).
            t: Current time.
            Omega: Rotational speed (rad/s).
            copy_to_bus: If True (default), copy results to DataBus for
                OutputDispatcher.  Set False in the hot loop to avoid
                per-step allocation — copies are only needed on output steps.

        Returns:
            pen:  (n_nodes,) penetration depth per node (0 = no contact)
            coords: (n_nodes, 5) --- [theta, x, yc, zc, r]
            interp: (n_nodes, 9) or (n_nodes, 21) --- [h_loc, ep_loc, alpha_loc, i_t, i_x, w00..]
        """
        u_p_arr = np.asarray(u_p, dtype=np.float64)
        if not u_p_arr.flags["C_CONTIGUOUS"]:
            u_p_arr = np.ascontiguousarray(u_p_arr)

        # Step 1: kinematics (writes _coords_buf: theta, x, yc, zc, r)
        self._kinematics_fn(
            u_p_arr, self._theta0, self._x0, self._y0, self._z0,
            self._dof_idx, float(Omega), float(t), self._coords_buf)

        # Compute casing radius per node
        # Fast path: JIT bilinear interpolation of pre-computed R_grid
        if (self._radius_kernel is not None and self._R_grid is not None
                and self._R_grid_params is not None):
            self._radius_kernel(
                self._coords_buf, self._R_grid, self._R_grid_params,
                self._R_buf)
            R_tip = self._R_buf
        else:
            R_tip = self._compute_R_tip(u_p_arr, float(t), float(Omega))

        # Step 2: interpolate coating state
        coating_h = self._db.get("coating.h")
        coating_ep = self._db.get("coating.ep")
        coating_alpha = self._db.get("coating.alpha")
        if coating_h is None:
            coating_h = np.zeros((1, 1), dtype=np.float64)
            coating_ep = np.zeros((1, 1), dtype=np.float64)
            coating_alpha = np.zeros((1, 1), dtype=np.float64)
        self._interpolate_fn(
            self._coords_buf, coating_h, coating_ep, coating_alpha,
            self._grid_params, self._interp_buf)

        # Step 3: gap detection
        self._gap_fn(self._coords_buf, self._interp_buf, R_tip, self._pen_buf)

        # Store penetration on DataBus for OutputDispatcher
        if copy_to_bus:
            self._db.set("penetration", self._pen_buf.copy())
            self._db.set("contacts_coords", self._coords_buf.copy())
            self._db.set("contacts_interp", self._interp_buf.copy())

        return self._pen_buf, self._coords_buf, self._interp_buf

    def penetration_at_nodes(self, u_p, t, Omega) -> np.ndarray:
        """Return penetration delta per tip node."""
        pen, _, _ = self.detect(u_p, t, Omega)
        return pen

    def snapshot_for_output(self) -> None:
        """Copy current detect buffers to DataBus for OutputDispatcher.

        Called only on output steps to avoid per-step allocation in the
        hot loop.  Must be called after detect() before the next detect()
        overwrites the buffers.
        """
        self._db.set("penetration", self._pen_buf.copy())
        self._db.set("contacts_coords", self._coords_buf.copy())
        self._db.set("contacts_interp", self._interp_buf.copy())

    def contacts_as_list(self, u_p, t, Omega):
        """Return list of contact dicts (backward-compatible format).

        .. deprecated::
            Use :meth:`detect` directly and pass raw arrays to
            :meth:`ForceAssembler.assemble`.  The list-of-dicts format
            is retained for testing and debugging only.

        Each dict: i_node, delta, h_loc, ep_loc, alpha_loc,
                   i_theta, i_x, w00..w11, yc, zc, r, v_rel
        """
        pen, coords, interp = self.detect(u_p, t, Omega)
        contacts = []
        Omega_f = float(Omega)
        for i in range(self._n_nodes):
            if pen[i] > 0.0:
                if self._interpolator_type == "BICUBIC_BSPLINE":
                    contacts.append({
                        "i_node": i,
                        "delta": pen[i],
                        "h_loc": interp[i, 0],
                        "ep_loc": interp[i, 1],
                        "alpha_loc": interp[i, 2],
                        "i_theta": int(interp[i, 3]),
                        "i_x": int(interp[i, 4]),
                        "w00": interp[i, 5], "w10": interp[i, 6],
                        "w20": interp[i, 7], "w30": interp[i, 8],
                        "w01": interp[i, 9], "w11": interp[i, 10],
                        "w21": interp[i, 11], "w31": interp[i, 12],
                        "w02": interp[i, 13], "w12": interp[i, 14],
                        "w22": interp[i, 15], "w32": interp[i, 16],
                        "w03": interp[i, 17], "w13": interp[i, 18],
                        "w23": interp[i, 19], "w33": interp[i, 20],
                        "yc": coords[i, 2], "zc": coords[i, 3], "r": coords[i, 4],
                        "v_rel": Omega_f * coords[i, 4],
                    })
                else:
                    contacts.append({
                        "i_node": i,
                        "delta": pen[i],
                        "h_loc": interp[i, 0],
                        "ep_loc": interp[i, 1],
                        "alpha_loc": interp[i, 2],
                        "i_theta": int(interp[i, 3]),
                        "i_x": int(interp[i, 4]),
                        "w00": interp[i, 5], "w10": interp[i, 6],
                        "w01": interp[i, 7], "w11": interp[i, 8],
                        "yc": coords[i, 2], "zc": coords[i, 3], "r": coords[i, 4],
                        "v_rel": Omega_f * coords[i, 4],
                    })
        return contacts


    def get_pipeline(self):
        """detect() as pipeline: (u_p, t, Omega) -> (pen, coords, interp)."""
        def _pipe(u_p, t, Omega):
            return self.detect(u_p, t, Omega, copy_to_bus=False)
        return _pipe


# Register for auto-discovery via ComponentRegistry
components.register_class("CONTACT_DETECTOR", ContactDetector)
