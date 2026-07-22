"""Compatibility shim — re-exports from the new rubimpact.kernels package.

All JIT kernels have been moved to rubimpact.kernels/.  This file exists
so that existing imports (like model_assembler) don't break during the
transition.  Importing this package triggers all kernel registrations.
"""
import rubimpact.kernels  # noqa: F401 — triggers all kernel registration
