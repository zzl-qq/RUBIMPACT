"""L2 modules — protocol-driven, pipeline-capable."""
from rubimpact.modules import contact_force

try:
    from rubimpact.modules import friction_force
except ImportError:
    friction_force = None  # type: ignore[assignment]

try:
    from rubimpact.modules import contact_detector
except ImportError:
    contact_detector = None  # type: ignore[assignment]

try:
    from rubimpact.modules import time_integrator
except ImportError:
    time_integrator = None  # type: ignore[assignment]

try:
    from rubimpact.modules import dynamic_relaxation
except ImportError:
    dynamic_relaxation = None  # type: ignore[assignment]
