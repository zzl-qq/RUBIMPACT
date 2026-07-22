"""Protocol-driven JIT pipeline factory.

Given a set of Module instances, reads their PipelineProtocol declarations
and auto-generates a single @njit function that loops over all nodes,
calling each stage's kernel in topological order.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from numba import njit
from rubimpact.core.registry import components


class PipelineFactory:
    """Builds composed @njit pipelines from module protocol declarations."""

    @classmethod
    def build(cls, modules: dict[str, object],
              protocol: str,
              shared_kernels: dict[str, str] | None = None,
              mode: str | None = None,
              extra_params: dict[str, float] | None = None) -> callable:
        """Build a composed @njit pipeline function.

        Args:
            modules: Dict mapping slot_name → Module instance.
            protocol: Orchestrator protocol name (e.g. "ForceAssembler").
            shared_kernels: Dict mapping shared stage name → "category/TYPE"
                           (e.g. {"geometry_decompose": "shared/GEOMETRY_DECOMPOSE"}).
            mode: Optional build mode ("readonly_coating" disables wear stages).
            extra_params: Optional extra params injected into the pipeline
                         (e.g. {"k_penalty": 0.0} for PENALTY fallback).

        Returns:
            A @njit compiled pipeline function with signature:
            pipeline(pen, coords, interp, Omega, dof_idx, F_total, F_normal, F_friction)
        """
        # Collect all stages from all modules
        all_stages: list[dict] = []
        stage_params: dict[str, float] = {}

        for slot_name, module in modules.items():
            proto = module.get_pipeline_protocol()
            if proto is None:
                continue
            for stage in proto.stages:
                # Skip optional stages based on mode
                if mode == "readonly_coating" and stage.name in ("apply_wear", "accumulate_stress"):
                    continue
                stage_info = {
                    "name": stage.name,
                    "kernel_ref": stage.kernel_ref,
                    "depends_on": stage.depends_on,
                    "optional": stage.optional,
                }
                all_stages.append(stage_info)
            stage_params.update(proto.params)

        # Merge extra params (supplied externally, e.g. k_penalty for PENALTY fallback)
        if extra_params:
            stage_params.update(extra_params)

        # Topological sort stages by depends_on
        sorted_stages = cls._topo_sort(all_stages)

        # Resolve kernel functions
        stage_fns = []
        for s in sorted_stages:
            kernel = cls._resolve_kernel(s["kernel_ref"]) if s["kernel_ref"] else None
            stage_fns.append((s["name"], kernel))

        # Compile pipeline
        return cls._compile_pipeline(stage_fns, stage_params)

    @staticmethod
    def _topo_sort(stages: list[dict]) -> list[dict]:
        """Topological sort stages by depends_on using Kahn's algorithm."""
        name_to_idx = {s["name"]: i for i, s in enumerate(stages)}
        in_degree = {s["name"]: 0 for s in stages}
        adj = {s["name"]: [] for s in stages}

        for s in stages:
            for dep in s.get("depends_on", []):
                if dep in name_to_idx:
                    in_degree[s["name"]] += 1
                    adj.setdefault(dep, []).append(s["name"])

        queue = deque([s["name"] for s in stages if in_degree[s["name"]] == 0])
        result = []
        while queue:
            current = queue.popleft()
            result.append(stages[name_to_idx[current]])
            for neighbor in adj.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(stages):
            raise ValueError("Cycle detected in pipeline stages")
        return result

    @staticmethod
    def _resolve_kernel(kernel_ref: str) -> callable:
        """Resolve a kernel reference "category/TYPE" to the actual @njit function."""
        if "/" not in kernel_ref:
            raise ValueError(f"Invalid kernel_ref: {kernel_ref}, expected 'category/TYPE'")
        category, type_name = kernel_ref.split("/", 1)
        spec = components.resolve_kernel(category, type_name)
        if spec is None:
            registered = components.list_types(category)
            raise ValueError(
                f"Unknown kernel: {kernel_ref}. Registered in '{category}': {registered}")
        return spec.fn

    @staticmethod
    def _compile_pipeline(stage_fns: list[tuple[str, callable]],
                          params: dict[str, float]) -> callable:
        """Compile stages into a single @njit pipeline function.

        For the ForceAssembler protocol, generates:
            pipeline(pen, coords, interp, Omega, dof_idx, F_total, F_normal, F_friction)

        Stage function signatures (per-node, scalar):
            normal_force_kernel(delta, h_loc, ep_loc, alpha_loc, params, out)
            friction_kernel(F_n, v_rel, params, out)
            geometry_kernel(Fy_n_out, Fz_n_out, Fy_t_out, Fz_t_out,
                           F_n, F_t, yc, zc, r_val)
            rom_map_kernel(F_total, F_normal, F_friction,
                          Fy_n, Fz_n, Fy_t, Fz_t,
                          ky, kz, has_normal, has_friction)
        """
        # Classify stage functions by name
        normal_fn = None
        friction_fn = None
        wear_fn = None
        for name, fn in stage_fns:
            if name == "compute_normal_force":
                normal_fn = fn
            elif name == "compute_friction_force":
                friction_fn = fn
            elif name == "apply_wear":
                wear_fn = fn

        # Build normal_params only when a normal-force kernel exists.
        # Use "pcl_params" if present; otherwise construct from E / A_cell.
        if normal_fn is not None:
            if "pcl_params" in params:
                normal_params = params["pcl_params"]
            else:
                normal_params = np.array(
                    [params["E"], params["A_cell"], 0.0, 0.0], dtype=np.float64)
        else:
            normal_params = np.zeros(4, dtype=np.float64)

        # Build friction_params only when a friction kernel exists.
        if friction_fn is not None:
            if "fric_params" in params:
                friction_params = params["fric_params"]
            else:
                friction_params = np.array([params["mu"]], dtype=np.float64)
        else:
            friction_params = np.zeros(1, dtype=np.float64)

        # ── Wear support ──
        has_wear = wear_fn is not None
        if has_wear:
            if "wear_params" not in params:
                raise ValueError(
                    "Pipeline has wear stage but 'wear_params' not in protocol. "
                    "Ensure ContactForceModule provides wear_params in its "
                    "PipelineProtocol.")
            wear_params = np.asarray(params["wear_params"], dtype=np.float64)

        # ── Cell area for PCL force scaling (sigma → force) ──
        # Required when a normal-force kernel exists; unused on PENALTY-only path.
        if normal_fn is not None:
            if "A_cell" not in params:
                raise ValueError(
                    "Pipeline has normal-force kernel but 'A_cell' not "
                    "in protocol params.")
            A_cell = float(params["A_cell"])
        else:
            A_cell = 1.0  # unused, only captured for Numba closure

        # PENALTY fallback: used only when no normal kernel is provided.
        if normal_fn is None:
            k_penalty_val = params["k_penalty"]
        else:
            k_penalty_val = 0.0

        @njit(fastmath=True)
        def pipeline(pen, coords, interp, Omega, dof_idx,
                     F_total, F_normal, F_friction,
                     coating_h, coating_ep, coating_alpha):
            n_nodes = pen.shape[0]
            n_r = F_total.shape[0]
            has_normal = F_normal.size > 0
            has_friction = F_friction.size > 0

            # Pre-allocate per-node output buffers (allocated once, reused per iteration).
            out_normal_buf = np.zeros(4, dtype=np.float64)
            out_fric_buf = np.zeros(1, dtype=np.float64)

            for i in range(n_nodes):
                delta = pen[i]
                if delta <= 0.0:
                    continue

                # ── Normal force ──
                F_n = 0.0
                if normal_fn is not None:
                    h_loc = interp[i, 0]
                    ep_loc = interp[i, 1]
                    alpha_loc = interp[i, 2]
                    out_normal_buf[0] = 0.0
                    out_normal_buf[1] = 0.0
                    out_normal_buf[2] = 0.0
                    out_normal_buf[3] = 0.0
                    normal_fn(delta, h_loc, ep_loc, alpha_loc,
                              normal_params, out_normal_buf)
                    F_n = out_normal_buf[0] * A_cell  # stress → force
                else:
                    # Inline: F_n = k * delta (PENALTY)
                    F_n = k_penalty_val * delta

                if F_n <= 0.0:
                    continue

                # ── Wear (coating state write-back) ──
                if has_wear:
                    dgamma = out_normal_buf[1] - interp[i, 1]
                    if dgamma > 0.0:
                        i_theta = int(interp[i, 3])
                        i_x = int(interp[i, 4])
                        # unified kernel: weights = interp[i, 5:]
                        wear_fn(i_theta, i_x, interp[i, 5:], dgamma,
                                coating_h, coating_ep, coating_alpha,
                                wear_params)

                # ── Friction force ──
                F_t = 0.0
                r_val = coords[i, 4]
                if friction_fn is not None:
                    v_rel = Omega * r_val
                    out_fric_buf[0] = 0.0
                    friction_fn(F_n, v_rel, friction_params, out_fric_buf)
                    F_t = out_fric_buf[0]

                # ── Geometry decomposition ──
                yc = coords[i, 2]
                zc = coords[i, 3]
                inv_r = 1.0 / max(r_val, 1e-12)
                Fy_n = F_n * yc * inv_r
                Fz_n = F_n * zc * inv_r
                Fy_t = -F_t * zc * inv_r
                Fz_t = F_t * yc * inv_r

                # ── ROM DOF mapping ──
                ky = dof_idx[i, 1]
                kz = dof_idx[i, 2]
                if 0 <= ky < n_r:
                    if has_normal:
                        F_normal[ky] += Fy_n
                    if has_friction:
                        F_friction[ky] += Fy_t
                    F_total[ky] += Fy_n + Fy_t
                if 0 <= kz < n_r:
                    if has_normal:
                        F_normal[kz] += Fz_n
                    if has_friction:
                        F_friction[kz] += Fz_t
                    F_total[kz] += Fz_n + Fz_t

        return pipeline
