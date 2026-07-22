"""L3 scalar JIT kernels — pure functions, no heap allocation.

Import order by dependency chain:
  kinematics → casing_radius → interpolator → gap
  constitutive → friction → wear
  predictor → corrector
  shared (geometry + ROM mapping)
"""

_kernel_modules = [
    "kinematics",
    "casing",
    "interpolator",
    "gap",
    "constitutive",
    "friction",
    "wear",
    "predictor",
    "corrector",
    "shared",
]

for _mod in _kernel_modules:
    try:
        __import__(f"rubimpact.kernels.{_mod}")
    except ImportError:
        pass  # module not yet created — added in its own task

del _mod, _kernel_modules
