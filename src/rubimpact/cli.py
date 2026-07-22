"""CLI entry point (framework S7.1).

Abaqus-style command interface::

    rmbopcl submit <input.inp>               Run simulation (1 core default)
    rmbopcl submit <input.inp> --cores 8     Run with 8 MKL threads
    rmbopcl submit <input.inp> --profile     Run + write performance profile
    rmbopcl check <input.inp>                Validate input file
    rmbopcl info <input.inp>                 Print model summary
    rmbopcl version                          Show version

Set environment variable RMBOPCL_CORES to override the default core count
for all runs (``--cores`` takes precedence).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from rubimpact.orchestrators.model_assembler import ModelAssembler

__version__ = "1.0.0"


def _physical_cpu_count() -> int:
    """Return number of physical CPU cores (not logical processors).

    For CPU-intensive dense linear algebra (MKL PARDISO), the optimal
    thread count is bounded by physical cores — hyperthreading on the
    same core shares the FPU and L1/L2 cache, often hurting throughput.

    Falls back to ``os.cpu_count()`` if detection fails.
    """
    try:
        if sys.platform == "win32":
            # PowerShell: reliable, available on all supported Windows
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).NumberOfCores"],
                text=True, stderr=subprocess.DEVNULL,
            )
            val = out.strip()
            if val.isdigit():
                return int(val)
            # Fallback: wmic (older Windows)
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "NumberOfCores"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for line in out.strip().splitlines():
                if line.strip().isdigit():
                    return int(line.strip())
        else:
            # Linux / macOS: os.sched_getaffinity (Python ≥3.14 friendly)
            try:
                return len(os.sched_getaffinity(0))
            except (AttributeError, NotImplementedError):
                # Parse /proc/cpuinfo (Linux)
                try:
                    with open("/proc/cpuinfo") as fh:
                        seen = set()
                        for line in fh:
                            if line.startswith("cpu cores"):
                                seen.add(int(line.split(":")[1].strip()))
                    if seen:
                        return sum(seen)
                except Exception:
                    pass
    except Exception:
        pass
    # Last resort: logical count (over-estimate is safe — user can tune down)
    return os.cpu_count() or 1


def _set_mkl_cores(cores: int) -> int:
    """Override MKL/OMP thread count at runtime (before pypardiso init).

    Returns the effective core count after hardware-cap validation.
    Caps at physical cores (not logical processors), since MKL dense
    linear algebra rarely benefits from hyperthreading.
    """
    phys_cores = _physical_cpu_count()
    logical = os.cpu_count() or phys_cores
    if cores > phys_cores:
        print(f"  NOTE: --cores {cores} > {phys_cores} physical cores "
              f"({logical} logical); using {phys_cores}")
        cores = phys_cores
    for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[var] = str(cores)
    return cores


def cmd_submit(args: argparse.Namespace) -> None:
    inp_path = args.input
    if not Path(inp_path).exists():
        print(f"ERROR: Input file not found: {inp_path}")
        sys.exit(1)

    # Resolve core count: CLI flag > env var > default 1
    cores = args.cores if args.cores > 0 else int(
        os.environ.get("RMBOPCL_CORES", "1"))
    phys_cores = _physical_cpu_count()
    logical = os.cpu_count() or phys_cores
    cores = _set_mkl_cores(cores)

    t_start = time.perf_counter()
    inp_abs = Path(inp_path).resolve()
    inp_dir = str(inp_abs.parent)
    print(f"RUBIMPACT v{__version__} — Processing: {inp_abs}")
    print(f"  MKL threads: {cores}  (physical cores: {phys_cores}, "
          f"logical processors: {logical})")
    try:
        assembler = ModelAssembler(base_dir=inp_dir)
        assembler.build_and_run(str(inp_abs), profile=args.profile)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.perf_counter() - t_start
    print(f"Total wall time: {elapsed:.1f} s")


def cmd_check(args: argparse.Namespace) -> None:
    """Validate input file without running simulation."""
    from rubimpact.orchestrators.model_assembler import KeywordParser

    inp_path = args.input
    if not Path(inp_path).exists():
        print(f"ERROR: Input file not found: {inp_path}")
        sys.exit(1)

    print(f"RUBIMPACT v{__version__} — Validating: {inp_path}")
    inp_text = Path(inp_path).read_text(encoding="utf-8")
    config = KeywordParser().parse(inp_text)

    print("Validation PASSED — input file is well-formed.")
    _cmd_info_from_config(config)


def cmd_info(args: argparse.Namespace) -> None:
    """Print model summary from input file."""
    from rubimpact.orchestrators.model_assembler import KeywordParser

    inp_path = args.input
    if not Path(inp_path).exists():
        print(f"ERROR: Input file not found: {inp_path}")
        sys.exit(1)

    inp_text = Path(inp_path).read_text(encoding="utf-8")
    config = KeywordParser().parse(inp_text)
    _cmd_info_from_config(config)


def _cmd_info_from_config(config: dict) -> None:
    """Print structured model summary."""
    model = config.get("MODEL", {})
    print(f"\n  Model: {model.get('NAME', '(unnamed)')}")

    # STEP parameters
    for i, step in enumerate(config.get("STEP", [])):
        sp = step.get("params", {})
        print(f"  STEP {i + 1}:")
        print(f"    Omega = {sp.get('Omega', '?')} rad/s")
        print(f"    h     = {sp.get('h', '?')} s")
        print(f"    T_f   = {sp.get('T_f', '?')} s")
        steps = int(round(float(sp.get("T_f", "0")) / float(sp.get("h", "1e-8"))))
        print(f"    steps ≈ {steps:,}")

    # ROM
    if "ROM" in config:
        rom = config["ROM"][0]
        print(f"  ROM: enabled (n_modal={rom.get('n_modal', '?')})")
    else:
        print(f"  ROM: disabled (full-order)")

    # Keywords present
    kw_list = [k for k in config if k not in ("MODEL", "STEP")]
    print(f"  Keywords: {', '.join(kw_list)}")

    # External data summary
    if "EXTERNAL_DATA" in config:
        ext = config["EXTERNAL_DATA"][0]
        for slot, sub in ext.get("sub_list", []):
            fname = sub.get("FILE", "?")
            print(f"  External: {slot} → {fname}")

    print()


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"RUBIMPACT v{__version__}")
    print("Rotating Machinery Blade-Off / Blade-Casing Rub-Impact Simulation")
    print("  Craig-Bampton ROM | Numba JIT | HDF5 Output | PARDISO (MKL)")
    print(f"  Python: {sys.version}")


def main():
    parser = argparse.ArgumentParser(
        prog="rmbopcl",
        description="RUBIMPACT — Rotating Machinery Rub-Impact Simulation",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- submit ---
    p_submit = sub.add_parser("submit", help="Submit a simulation job")
    p_submit.add_argument("input", help="Input file (.inp)")
    p_submit.add_argument(
        "--profile", action="store_true",
        help="Write performance profile (job.profile) after execution")
    p_submit.add_argument(
        "--cores", type=int, default=1,
        help="MKL/OMP thread count (1 = single-core; capped at logical CPU count)")
    p_submit.set_defaults(func=cmd_submit)

    # --- check ---
    p_check = sub.add_parser("check", help="Validate input file without running")
    p_check.add_argument("input", help="Input file (.inp)")
    p_check.set_defaults(func=cmd_check)

    # --- info ---
    p_info = sub.add_parser("info", help="Print model summary")
    p_info.add_argument("input", help="Input file (.inp)")
    p_info.set_defaults(func=cmd_info)

    # --- version ---
    p_version = sub.add_parser("version", help="Show version information")
    p_version.set_defaults(func=cmd_version)

    # --- help (default) ---
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
