"""ModelAssembler — unified configure() loop, zero special branches.

All module discovery, JIT dispatch, pipeline scheduling, and output are
driven by the registries.  Adding a new module type requires only
registration + configure() — no assembler changes.
"""
import gc
import time
from pathlib import Path

import numpy as np

from rubimpact.infra.databus import DataBus
from rubimpact.infra.hdf5_writer import HDF5Writer
from rubimpact.infra.job_manager import JobManager
from rubimpact.infra.state_manager import StateManager
from rubimpact.infra.output_catalog import OutputDispatcher, output_catalog
from rubimpact.core.registry import components
from rubimpact.core.scheduler import scheduler

# Import JIT kernels to trigger registration
import rubimpact.kernels  # noqa: F401 — triggers kernel registration

# Import all module classes to trigger old-registry registration
import rubimpact.init.external_data   # noqa: F401
import rubimpact.init.casing          # noqa: F401
import rubimpact.init.coating         # noqa: F401
import rubimpact.init.matrix_assembly # noqa: F401
import rubimpact.init.rom             # noqa: F401
import rubimpact.modules.contact_detector  # noqa: F401
import rubimpact.runtime.constitutive     # noqa: F401
import rubimpact.modules.time_integrator  # noqa: F401
import rubimpact.modules.contact_force    # noqa: F401
import rubimpact.modules.dynamic_relaxation  # noqa: F401
import rubimpact.orchestrators.force_assembler  # noqa: F401

DOF_PER_NODE = 3
CONSTRAINED_DIAG = 1e30


# ═══════════════════════════════════════════════════════════════════
# KeywordParser — unchanged from original model_assembler.py
# ═══════════════════════════════════════════════════════════════════

class KeywordParser:
    """Parse INP keyword syntax with *SUBKEYWORD indentation nesting.

    Top-level keywords (indent=0):
        *KEYWORD, TYPE=TYPE, param=value
          *SUBKEYWORD, TYPE=TYPE             <- indent >= 4
          sub_param=value
    """

    def parse(self, text: str) -> dict:
        config: dict = {}
        current_entry = None
        current_sub_entry = None
        in_step = False
        current_output = None

        for raw in text.split("\n"):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw) - len(raw.lstrip(" "))

            if stripped.startswith("*"):
                upper = stripped.upper()
                if upper.startswith("*END MODEL"):
                    break
                if upper.startswith("*END STEP"):
                    in_step = False
                    current_output = None
                    continue
                if upper.startswith("*OUTPUT"):
                    _, params = self._parse_keyword_line(stripped)
                    current_output = {**params, "variables": []}
                    if in_step and config.get("STEP"):
                        config["STEP"][-1]["outputs"].append(current_output)
                    continue

                keyword, params = self._parse_keyword_line(stripped)
                current_output = None

                if keyword == "MODEL":
                    config["MODEL"] = params
                    current_entry = None
                    in_step = False
                    continue
                if keyword == "STEP":
                    entry = {**params, "params": {}, "outputs": []}
                    config.setdefault("STEP", []).append(entry)
                    current_entry = entry
                    in_step = True
                elif indent >= 4 and current_entry is not None:
                    # Submodule keyword (*HARDENING, *WEAR_LAW, ...)
                    slot = keyword.lower()  # normalize: *HARDENING -> hardening
                    sub_entry = {**params, "submodules": {}}
                    if current_sub_entry is not None and indent >= 8:
                        current_sub_entry["submodules"][slot] = sub_entry
                    else:
                        current_entry.setdefault("submodules", {})[slot] = sub_entry
                        current_entry.setdefault("sub_list", []).append(
                            (slot, sub_entry))
                    current_sub_entry = sub_entry
                else:
                    entry = {**params, "submodules": {}, "sub_list": []}
                    in_step = False
                    config.setdefault(keyword, []).append(entry)
                    current_entry = entry
                    current_sub_entry = None
                continue

            # Data lines
            if in_step:
                step = config["STEP"][-1]
                if current_output is not None and "=" not in stripped:
                    var = stripped.strip(", ")
                    if var:
                        current_output["variables"].append(var)
                elif "=" in stripped:
                    step["params"].update(self._parse_params(stripped))
                continue

            if current_entry is None:
                continue

            if "=" in (first_token := stripped.split(",", 1)[0]):
                params = self._parse_params(stripped)
                if indent >= 4 and current_sub_entry is not None:
                    current_sub_entry.update(params)
                else:
                    current_entry.update(params)

        return config

    def _parse_keyword_line(self, line: str):
        line = line.lstrip("*")
        if "," in line:
            kw, rest = line.split(",", 1)
            return kw.strip().upper(), self._parse_params(rest)
        return line.strip().upper(), {}

    @staticmethod
    def _parse_params(params_str: str) -> dict:
        result = {}
        for part in params_str.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                result[k.strip()] = v.strip()
        return result


# ═══════════════════════════════════════════════════════════════════
# Scheduler setup — register pipeline phases (topological ordering)
# ═══════════════════════════════════════════════════════════════════
# All module classes now inherit from Module directly (no monkey-patching).
# Each module file self-registers via components.register_class().
# get_pipeline() is a proper method on each runtime class.

def _setup_scheduler():
    """Register execution-phase ordering on the global scheduler.

    COATING and CONSTITUTIVE are optional — penalty-contact models
    and non-wearing analyses don't need them.
    """
    _phases = {
        ("INIT", "EXTERNAL_DATA"):    {"after": [], "required": True},
        ("INIT", "CASING"):           {"after": ["EXTERNAL_DATA"], "required": True},
        ("INIT", "COATING"):          {"after": ["CASING"], "required": False},
        ("INIT", "MATRIX_ASSEMBLY"):  {"after": ["EXTERNAL_DATA"], "required": True},
        ("INIT", "ROM"):              {"after": ["MATRIX_ASSEMBLY", "COATING"],
                                       "required": False},
        ("RUNTIME", "CONTACT_DETECTOR"):  {"after": [], "required": True},
        ("RUNTIME", "CONSTITUTIVE"):  {"after": ["COATING"], "required": False},
        ("RUNTIME", "TIME_INTEGRATOR"): {"after": ["ROM"], "required": True},
        ("RUNTIME", "FORCE_ASSEMBLER"): {"after": ["CONTACT_DETECTOR",
                                                     "CONSTITUTIVE"],
                                          "required": True},
        ("RUNTIME", "DYNAMIC_RELAXATION"): {"after": [], "required": False},
    }
    for (phase, cat), info in _phases.items():
        scheduler.register_phase(
            phase, cat, after=info["after"], required=info["required"])


_setup_scheduler()


# ═══════════════════════════════════════════════════════════════════
# ModelAssembler — zero special branches
# ═══════════════════════════════════════════════════════════════════

class ModelAssembler:
    """Top-level orchestrator — unified configure() loop, zero special cases.

    All module instantiation and configuration flows through a single
    configure() call per phase entry.  No if/elif branching on
    entry.category.
    """

    def __init__(self, base_dir="."):
        self.db = DataBus()
        self.job = JobManager(base_dir)

    def build_and_run(self, inp_path: str, profile: bool = False) -> None:
        """Parse INP, configure all modules, run time loop.

        ZERO if/elif branches based on entry.category.upper().
        """
        inp_text = Path(inp_path).read_text(encoding="utf-8")
        config = KeywordParser().parse(inp_text)

        # ── Shared context — every module gets this ──
        step0 = config["STEP"][0]
        sp = step0["params"]
        context = {
            "inp_dir": str(Path(inp_path).resolve().parent),
            "Omega": float(sp["Omega"]),
            "h": float(sp["h"]),
            "T_f": float(sp["T_f"]),
        }

        model_name = config.get("MODEL", {}).get("NAME", "JOB")
        self.job.init(model_name)
        self.job.save_input_copy(inp_text)

        Omega, h, T_f = context["Omega"], context["h"], context["T_f"]

        # ── INIT: unified configure() loop ──
        for entry in scheduler.schedule(config, phase="INIT"):
            cls = components.resolve_class(entry.category)
            if cls is None:
                continue
            module = cls(self.db, context)
            module.configure(entry.cfg)

        # Post-INIT: pre-compute casing radius grid on coating grid
        if "COATING" in config and "CASING" in config:
            from rubimpact.init.casing import Casing
            Casing.build_R_grid(self.db)

        # Full-order fallback (no *ROM)
        if "ROM" not in config:
            self._full_order_fallback()

        # ── RUNTIME: unified configure() loop ──
        runtime: dict[str, object] = {}
        for entry in scheduler.schedule(config, phase="RUNTIME"):
            cls = components.resolve_class(entry.category)
            if cls is None:
                continue
            module = cls(self.db, context)
            module.configure(entry.cfg)
            runtime[entry.category] = module

        # ── Run time loop ──
        n_r = int(self.db.get("rom.n_r", self.db.get("matrices.mass").shape[0]))
        self._run_loop(runtime, config, n_r, Omega, h, T_f, profile=profile)

        self.job.cleanup()

    # ── Full-order fallback ──────────────────────────────────────────

    def _full_order_fallback(self) -> None:
        """No *ROM: use full-order matrices."""
        M = self.db.get("matrices.mass")
        K = self.db.get("matrices.K_omega")
        D = self.db.get("matrices.D_full")
        diag_K = K.diagonal()
        tip_nodes = self.db.get("nodes.tip", {})
        tip_dof_map = []
        for nid in sorted(tip_nodes.keys()):
            base = (nid - 1) * DOF_PER_NODE
            for d in range(DOF_PER_NODE):
                dof = base + d
                if dof < len(diag_K) and diag_K[dof] < CONSTRAINED_DIAG:
                    tip_dof_map.append(dof)
        self.db.set("rom.tip_dof_map", tip_dof_map)
        self.db.set("rom.enabled", False)
        self.db.set("rom.n_r", M.shape[0])
        self.db.set("rom.M_r", M.to_dense())
        self.db.set("rom.K_r", K.to_dense())
        self.db.set("rom.D_r", D.to_dense())

    # ── Time loop ────────────────────────────────────────────────────

    def _run_loop(self, runtime: dict, config: dict,
                  n_r: int, Omega: float, h: float, T_f: float,
                  profile: bool = False) -> None:
        """Main time-stepping loop — pipeline-driven, no module-type dispatch."""
        state = StateManager(n_r)
        writer = HDF5Writer(self.job.output_dir)

        # ── Dynamic Relaxation (optional) ──
        if "DYNAMIC_RELAXATION" in config:
            dr_module = runtime.get("DYNAMIC_RELAXATION")
            cd = runtime["CONTACT_DETECTOR"]
            fa = runtime["FORCE_ASSEMBLER"]
            M_r = self.db.get("rom.M_r")
            K_r = self.db.get("rom.K_r")
            ch = self.db.get("coating.h")
            cep = self.db.get("coating.ep")
            calpha = self.db.get("coating.alpha")
            u_init, dr_ok, _ = dr_module.run(
                cd.get_pipeline(), fa.get_pipeline(), np.zeros(n_r),
                K_r, Omega, coating_h=ch, coating_ep=cep,
                coating_alpha=calpha)
            if u_init is not None:
                state.u_n = u_init.copy()
                state.u_nm1 = u_init.copy()
            if not dr_ok:
                print("  [WARN] DR did not converge")

        # ── Output setup ──
        od = OutputDispatcher()
        od.initialize(writer, self.db)

        outputs = config["STEP"][0].get("outputs", [])
        if not outputs:
            raise ValueError("*STEP requires at least one *OUTPUT")

        hist_freq, hist_vars, field_freq, field_vars = \
            self._parse_outputs(outputs)

        # Pre-open history datasets
        for var in hist_vars:
            entry = output_catalog.get(var)
            if entry:
                od.open_history(var, entry["ndim"])

        # ── Pipeline callables ──
        ti = runtime["TIME_INTEGRATOR"]
        cd = runtime["CONTACT_DETECTOR"]
        fa = runtime["FORCE_ASSEMBLER"]

        detect_pipe = cd.get_pipeline()
        assemble_pipe = fa.get_pipeline()

        ti.initialize(h)

        num_steps = int(round(T_f / h))
        print(f"Starting: {num_steps:,} steps, h={h:.1e}, T_f={T_f}")

        # .sta / .msg progress files
        out_dir = self.job.output_dir
        sta_path = out_dir / "job.sta"
        msg_path = out_dir / "job.msg"
        _sta = open(sta_path, "w", encoding="utf-8")
        _sta.write(f"RUBIMPACT JOB STATUS\n"
                   f"Steps: {num_steps:,}  h={h:.2e}  T_f={T_f}\n"
                   f"{'='*60}\n")

        def _log_msg(level: str, text: str) -> None:
            with open(msg_path, "a", encoding="utf-8") as mf:
                mf.write(f"[{level}] {text}\n")

        report_interval = max(1, min(50000, num_steps // 200))
        step_times = [] if profile else None

        gc.disable()
        t_wall_start = time.perf_counter()
        try:
            for step_idx in range(num_steps):
                t = step_idx * h
                t_step_start = time.perf_counter() if profile else 0.0

                # Predict
                u_p = ti.predict(state.u_n, state.u_nm1)

                # Detect + Assemble forces via pipelines
                pen, coords, interp = detect_pipe(u_p, t, Omega)
                # Request force components needed for output (F_normal, F_friction)
                _requested = set()
                if "CFN" in hist_vars:
                    _requested.add("F_normal")
                if "CFT" in hist_vars:
                    _requested.add("F_friction")
                F_total = assemble_pipe(pen, coords, interp, Omega,
                                        requested=_requested if _requested else None)
                self.db.set("current_h", h)

                # Correct
                u_new = ti.correct(u_p, F_total)

                # Advance
                state.advance(u_new, h)

                # Output: snapshot detect buffers only on output steps
                if (hist_freq > 0 and step_idx % hist_freq == 0) or \
                   (field_freq > 0 and step_idx % field_freq == 0):
                    cd.snapshot_for_output()

                if hist_freq > 0 and step_idx % hist_freq == 0:
                    for var in hist_vars:
                        od.write_history(var, self.db, state)

                if field_freq > 0 and step_idx % field_freq == 0:
                    for var in field_vars:
                        od.write_field(var, self.db, step_idx)

                if profile:
                    step_times.append(time.perf_counter() - t_step_start)

                # Progress report
                if step_idx % report_interval == 0 or \
                   step_idx == num_steps - 1:
                    pct = 100.0 * step_idx / max(num_steps, 1)
                    elapsed = time.perf_counter() - t_wall_start
                    eta = (elapsed / (step_idx + 1) * (num_steps - step_idx - 1)
                           if step_idx > 0 else 0.0)
                    print(
                        f"\r  [{pct:5.1f}%] Step {step_idx:,}/{num_steps:,}"
                        f"  t={t:.6e}  Elapsed={elapsed:.0f}s  ETA={eta:.0f}s    ",
                        end="", flush=True)
                    _sta.write(
                        f"  STEP 1  INCREMENT {step_idx:8d}  "
                        f"TIME={t:.8e}  PROGRESS={pct:5.1f}%  "
                        f"WALL={elapsed:.1f}s  ETA={eta:.1f}s\n")
                    _sta.flush()
        finally:
            gc.enable()

        print()
        elapsed_total = time.perf_counter() - t_wall_start
        _sta.write(f"\nCOMPLETED. Total wall time: {elapsed_total:.1f} s\n")
        _sta.close()

        if profile:
            profile_path = out_dir / "job.profile"
            with open(profile_path, "w", encoding="utf-8") as pf:
                pf.write("# step_idx  wall_time_s\n")
                for idx, dt in enumerate(step_times):
                    pf.write(f"{idx:8d}  {dt:.9e}\n")

        writer.close()
        print(f"Complete. Output: {out_dir}")

    # ── Output parsing ───────────────────────────────────────────────

    @staticmethod
    def _parse_outputs(outputs: list) -> tuple:
        """Parse *OUTPUT blocks from step config.

        Returns: (hist_freq, hist_vars, field_freq, field_vars)
        """
        hist_freq, hist_vars = -1, []
        field_freq, field_vars = -1, []
        for out in outputs:
            otype = (out.get("TYPE") or "").upper()
            freq = int(out.get("FREQUENCY", "1"))
            vars_list = out.get("variables", [])
            if otype == "HISTORY":
                hist_freq = freq
                hist_vars = vars_list
            elif otype == "FIELD":
                field_freq = freq
                field_vars = vars_list
        return hist_freq, hist_vars, field_freq, field_vars
