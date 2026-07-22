"""Regression: all runtime kernels registered and callable."""
import numpy as np
import pytest
from rubimpact.core.registry import components

# Import kernel modules to trigger ComponentRegistry registration
import rubimpact.kernels.constitutive  # noqa: F401
import rubimpact.kernels.friction  # noqa: F401
import rubimpact.kernels.wear  # noqa: F401
import rubimpact.kernels.predictor  # noqa: F401
import rubimpact.kernels.corrector  # noqa: F401


def test_constitutive_kernel_registered():
    spec = components.resolve_kernel("constitutive", "PLASTIC_COATING_LAW")
    assert spec is not None
    assert spec.signature == "pcl_return_mapping"

    # Quick sanity call
    fn = spec.fn
    out = np.zeros(4, dtype=np.float64)
    # Y=3000 keeps the test elastic (sigma_trial=2000 < Y=3000)
    params = np.array([200e3, 3000.0, 1000.0, 1.0], dtype=np.float64)
    fn(0.01, 1.0, 0.0, 0.0, params, out)
    # Elastic: sigma = E * delta / h = 200e3 * 0.01 / 1.0 = 2000
    assert out[0] == pytest.approx(2000.0, rel=1e-10)
    assert out[1] == 0.0  # dgamma = 0 (no plastic flow at this load)


def test_friction_kernels_registered():
    assert components.resolve_kernel("friction", "COULOMB") is not None
    assert components.resolve_kernel("friction", "STRIBECK") is not None


def test_wear_kernel_registered():
    assert components.resolve_kernel("wear", "PLASTIC_STRAIN") is not None


def test_predictor_kernel_registered():
    assert components.resolve_kernel("predictor", "LINEAR") is not None


def test_corrector_kernel_registered():
    assert components.resolve_kernel("corrector", "CONTACT_CONSTRAINED") is not None
