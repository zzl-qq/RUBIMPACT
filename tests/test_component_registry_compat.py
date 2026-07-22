# tests/test_jit_registry.py — migrated to ComponentRegistry
from numba import njit
from rubimpact.core.registry import components, KernelSpec


def test_register_and_get():
    """注册后能正确取回。"""
    @njit
    def test_kernel(a):
        return a + 1.0

    spec = KernelSpec(fn=test_kernel, signature="test", stage="test")
    components.register("_test_cat", "TEST", spec)
    resolved = components.resolve("_test_cat", "TEST")

    # Numba JIT function
    result = resolved.fn(1.0)
    assert result == 2.0


def test_case_insensitive_lookup():
    """category 和 type_name 查找应大小写不敏感。"""
    @njit
    def kernel(x): return x * 2.0

    spec = KernelSpec(fn=kernel, signature="test_case", stage="test")
    components.register("CASE_TEST", "MIXED_CASE", spec)
    resolved = components.resolve("case_test", "mixed_case")
    assert resolved is not None
    assert resolved.fn(3.0) == 6.0


def test_missing_returns_none():
    """未注册的返回 None。"""
    fn = components.resolve("nonexistent", "STUFF")
    assert fn is None


def test_list_types():
    """list_types 返回已注册 TYPE 列表。"""
    @njit
    def self_contained_kernel(x): return x

    spec = KernelSpec(fn=self_contained_kernel, signature="sc", stage="lt")
    components.register("LIST_TEST", "SELF_CONTAINED", spec)
    types = components.list_types("LIST_TEST")
    assert "SELF_CONTAINED" in types

    types_empty = components.list_types("nonexistent")
    assert types_empty == []


def test_register_duplicate_overwrites():
    """重复注册覆盖旧函数。"""
    @njit
    def old_fn(x): return x
    @njit
    def new_fn(x): return x + 100.0

    components.register("test_dup", "DUP",
        KernelSpec(fn=old_fn, signature="old", stage="dup"))
    components.register("test_dup", "DUP",
        KernelSpec(fn=new_fn, signature="new", stage="dup"))
    fn = components.resolve("test_dup", "DUP")
    assert fn.fn(1.0) == 101.0


def test_signature_validation():
    """注册时检查是否为可调用对象。"""
    def not_numba(x):
        return x
    # 不抛异常 — 兼容 Python 函数
    components.register("fallback", "PYTHON",
        KernelSpec(fn=not_numba, signature="python", stage="fallback"))
    fn = components.resolve("fallback", "PYTHON")
    assert fn.fn(5.0) == 5.0

