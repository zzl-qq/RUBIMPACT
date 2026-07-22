"""Tests for ModuleDescriptor port validation and path inference."""

import pytest

from rubimpact.infra.module_descriptor import (
    ModuleDescriptor,
    ParamSpec,
    PortSpec,
    _infer_category_from_path,
    _types_compatible,
    discover_modules,
    validate_config,
    validate_ports,
)


# ═══════════════════════════════════════════════════════════════
# Path inference
# ═══════════════════════════════════════════════════════════════


def test_path_inference_from_directory():
    """Infer category from directory path."""
    inferred = _infer_category_from_path(
        "modules/CONSTITUTIVE/hardening/LINEAR_ISOTROPIC.yaml"
    )
    assert inferred == "CONSTITUTIVE/hardening"


def test_path_inference_single_segment():
    """Single segment after modules -> that segment is the category."""
    inferred = _infer_category_from_path("modules/CASING/LOBE.yaml")
    assert inferred == "CASING"


def test_path_inference_no_modules_segment():
    """If 'modules' not in path, returns empty string."""
    inferred = _infer_category_from_path("some/other/path/MODULE.yaml")
    assert inferred == ""


def test_path_inference_root_yaml():
    """YAML directly in modules/ (no subdirectory) returns empty string."""
    inferred = _infer_category_from_path("modules/ROOT.yaml")
    assert inferred == ""


# ═══════════════════════════════════════════════════════════════
# Type compatibility
# ═══════════════════════════════════════════════════════════════


def test_types_compatible_any():
    """Any on either side is always compatible."""
    assert _types_compatible("any", "scalar") is True
    assert _types_compatible("scalar", "any") is True
    assert _types_compatible("any", "any") is True


def test_types_compatible_exact_match():
    """Exact match is compatible."""
    assert _types_compatible("scalar", "scalar") is True
    assert _types_compatible("matrix", "matrix") is True
    assert _types_compatible("ndarray", "ndarray") is True


def test_types_compatible_mismatch():
    """Different non-any types are incompatible."""
    assert _types_compatible("scalar", "matrix") is False
    assert _types_compatible("matrix", "scalar") is False


def test_types_compatible_ndarray_vector():
    """ndarray can satisfy vector[n]."""
    assert _types_compatible("ndarray", "vector[3]") is True
    assert _types_compatible("ndarray", "vector[6]") is True


def test_types_compatible_ndarray_matrix_requires():
    """ndarray can satisfy matrix[mxn] in requires."""
    assert _types_compatible("ndarray", "matrix[3x3]") is True


# ═══════════════════════════════════════════════════════════════
# Port validation
# ═══════════════════════════════════════════════════════════════


def _lifecycle():
    """Shared lifecycle dict for port validation test modules."""
    return {"phase": "INIT", "required": True}


def test_port_validation_missing_producer():
    """Requires port but no module provides it -> error."""
    desc_a = ModuleDescriptor(
        id="MODULE_A", category="TEST_A",
        lifecycle=_lifecycle(),
        provides=[PortSpec(name="data.x", type="ndarray")]
    )
    desc_b = ModuleDescriptor(
        id="MODULE_B", category="TEST_B",
        lifecycle=_lifecycle(),
        requires=[PortSpec(name="data.y", type="ndarray")]  # no one provides
    )
    discovered = {
        "TEST_A/MODULE_A": desc_a,
        "TEST_B/MODULE_B": desc_b,
    }
    errors = validate_ports(discovered)
    assert any("data.y" in e for e in errors)
    assert any("no module provides" in e.lower() for e in errors)


def test_port_validation_type_mismatch():
    """Port type mismatch -> error."""
    desc_a = ModuleDescriptor(
        id="MODULE_A", category="TEST_A",
        lifecycle=_lifecycle(),
        provides=[PortSpec(name="data.z", type="scalar")]
    )
    desc_b = ModuleDescriptor(
        id="MODULE_B", category="TEST_B",
        lifecycle=_lifecycle(),
        requires=[PortSpec(name="data.z", type="matrix")]  # type conflict
    )
    discovered = {
        "TEST_A/MODULE_A": desc_a,
        "TEST_B/MODULE_B": desc_b,
    }
    errors = validate_ports(discovered)
    assert any("data.z" in e for e in errors)
    assert any("type" in e.lower() for e in errors)


def test_port_validation_compatible_types_pass():
    """Compatible types produce no errors."""
    desc_a = ModuleDescriptor(
        id="MODULE_A", category="TEST_A",
        lifecycle=_lifecycle(),
        provides=[PortSpec(name="data.w", type="scalar")]
    )
    desc_b = ModuleDescriptor(
        id="MODULE_B", category="TEST_B",
        lifecycle=_lifecycle(),
        requires=[PortSpec(name="data.w", type="scalar")]
    )
    discovered = {
        "TEST_A/MODULE_A": desc_a,
        "TEST_B/MODULE_B": desc_b,
    }
    errors = validate_ports(discovered)
    assert errors == []


def test_port_validation_dim_mismatch():
    """Dim mismatch produces an error."""
    desc_a = ModuleDescriptor(
        id="MODULE_A", category="TEST_A",
        lifecycle=_lifecycle(),
        provides=[PortSpec(name="data.v", type="vector[3]", dim=3)]
    )
    desc_b = ModuleDescriptor(
        id="MODULE_B", category="TEST_B",
        lifecycle=_lifecycle(),
        requires=[PortSpec(name="data.v", type="vector[6]", dim=6)]
    )
    discovered = {
        "TEST_A/MODULE_A": desc_a,
        "TEST_B/MODULE_B": desc_b,
    }
    errors = validate_ports(discovered)
    assert any("data.v" in e for e in errors)
    assert any("dim" in e.lower() for e in errors)


def test_port_validation_all_pass_no_ports():
    """Modules with no ports produce no errors."""
    desc_a = ModuleDescriptor(id="MODULE_A", category="TEST_A",
                              lifecycle=_lifecycle())
    desc_b = ModuleDescriptor(id="MODULE_B", category="TEST_B",
                              lifecycle=_lifecycle())
    discovered = {
        "TEST_A/MODULE_A": desc_a,
        "TEST_B/MODULE_B": desc_b,
    }
    errors = validate_ports(discovered)
    assert errors == []


def test_port_validation_any_type_compatible():
    """Any type is compatible with everything."""
    desc_a = ModuleDescriptor(
        id="MODULE_A", category="TEST_A",
        lifecycle=_lifecycle(),
        provides=[PortSpec(name="data.p", type="any")]
    )
    desc_b = ModuleDescriptor(
        id="MODULE_B", category="TEST_B",
        lifecycle=_lifecycle(),
        requires=[PortSpec(name="data.p", type="matrix[3x3]")]
    )
    discovered = {
        "TEST_A/MODULE_A": desc_a,
        "TEST_B/MODULE_B": desc_b,
    }
    errors = validate_ports(discovered)
    assert errors == []


# ═══════════════════════════════════════════════════════════════
# validate_config integration
# ═══════════════════════════════════════════════════════════════


def test_compatible_with_check_warning():
    """compatible_with not satisfied is non-blocking."""
    desc = ModuleDescriptor(
        id="MODULE_C", category="TEST_C",
        compatible_with=["OTHER_MODULE"],
        lifecycle={"required": False},
    )
    discovered = {"TEST_C/MODULE_C": desc}
    config = {"TEST_C": [{"TYPE": "MODULE_C"}]}
    errors = validate_config(discovered, config)
    # compatible_with 不匹配只产生非阻断提示，不阻断
    assert errors == []  # no blocking errors


def test_validate_config_includes_port_errors():
    """Port validation errors are now blocking (returned as errors).

    Only lifecycle-bearing modules participate in port validation.
    Submodules without lifecycle (damping, friction, hardening, …)
    are internal — their interfaces are defined by Python function
    signatures, not DataBus keys.
    """
    desc_a = ModuleDescriptor(
        id="MODULE_A", category="TEST_A",
        lifecycle={"phase": "INIT", "required": True},
        provides=[PortSpec(name="data.x", type="scalar")]
    )
    desc_b = ModuleDescriptor(
        id="MODULE_B", category="TEST_B",
        lifecycle={"phase": "INIT", "required": True},
        requires=[PortSpec(name="data.z", type="ndarray")]  # missing provider
    )
    discovered = {
        "TEST_A/MODULE_A": desc_a,
        "TEST_B/MODULE_B": desc_b,
    }
    config = {
        "TEST_A": [{"TYPE": "MODULE_A"}],
        "TEST_B": [{"TYPE": "MODULE_B"}],
    }
    errors = validate_config(discovered, config)
    # Port errors MUST be in the return value (blocking validation)
    assert any("data.z" in e for e in errors)
    assert any("no module provides" in e.lower() for e in errors)
