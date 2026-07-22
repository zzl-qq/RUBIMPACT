"""Tests for ComponentRegistry."""
from numba import njit
from rubimpact.core.registry import ComponentRegistry, KernelSpec, ModuleSpec


def test_register_and_resolve_kernel():
    """注册后能正确取回 KernelSpec。"""
    reg = ComponentRegistry()

    @njit
    def add_one(x):
        return x + 1.0

    spec = KernelSpec(fn=add_one, signature="add_one", stage="test_stage")
    reg.register("test_cat", "TEST_TYPE", spec)

    resolved = reg.resolve("test_cat", "TEST_TYPE")
    assert resolved is not None
    assert resolved.fn(1.0) == 2.0
    assert resolved.signature == "add_one"
    assert resolved.stage == "test_stage"


def test_register_and_resolve_module():
    """注册后能正确取回 ModuleSpec。"""
    reg = ComponentRegistry()

    def builder(db, ctx):
        return {"built": True}

    spec = ModuleSpec(builder=builder, protocol="TestProtocol", required=True)
    reg.register("contact_force", "PCL_CONTACT", spec)

    resolved = reg.resolve("contact_force", "PCL_CONTACT")
    assert resolved is not None
    assert resolved.protocol == "TestProtocol"
    assert resolved.required is True
    assert resolved.builder(None, {}) == {"built": True}


def test_case_insensitive_lookup():
    """category 和 type_name 查找大小写不敏感。"""
    reg = ComponentRegistry()

    @njit
    def kernel(x):
        return x * 2.0

    reg.register("FRICTION", "COULOMB", KernelSpec(fn=kernel, signature="coulomb", stage="friction"))
    fn = reg.resolve("friction", "coulomb")
    assert fn is not None
    assert fn.fn(3.0) == 6.0


def test_missing_returns_none():
    """未注册的返回 None。"""
    reg = ComponentRegistry()
    assert reg.resolve("nonexistent", "TEST") is None


def test_list_types():
    """list_types 返回某个 category 的所有 TYPE。"""
    reg = ComponentRegistry()

    @njit
    def k1(x): return x

    @njit
    def k2(x): return x

    reg.register("friction", "COULOMB", KernelSpec(fn=k1, signature="c", stage="f"))
    reg.register("friction", "STRIBECK", KernelSpec(fn=k2, signature="s", stage="f"))

    types_list = reg.list_types("friction")
    assert "COULOMB" in types_list
    assert "STRIBECK" in types_list


def test_list_combinations():
    """list_combinations 返回编排器的所有合法 TYPE 组合。"""
    reg = ComponentRegistry()

    @njit
    def dummy(x): return x

    reg.register("contact_force", "PCL_CONTACT",
        ModuleSpec(builder=lambda d, c: None, protocol="NormalForceProtocol"))
    reg.register("contact_force", "PENALTY",
        ModuleSpec(builder=lambda d, c: None, protocol="NormalForceProtocol"))
    reg.register("friction_force", "COULOMB",
        ModuleSpec(builder=lambda d, c: None, protocol="FrictionProtocol"))
    reg.register("friction_force", "STRIBECK",
        ModuleSpec(builder=lambda d, c: None, protocol="FrictionProtocol"))

    # Register orchestrator slots
    reg.register_orchestrator("ForceAssembler",
        slots=["contact_force", "friction_force"])

    combos = reg.list_combinations("ForceAssembler")
    assert len(combos) == 4  # 2 × 2
    expected = {
        ("PCL_CONTACT", "COULOMB"),
        ("PCL_CONTACT", "STRIBECK"),
        ("PENALTY", "COULOMB"),
        ("PENALTY", "STRIBECK"),
    }
    actual = {(c["contact_force"], c["friction_force"]) for c in combos}
    assert actual == expected
