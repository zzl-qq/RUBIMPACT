"""Tests for PipelineRegistry with Kahn topological sort."""

import pytest

from rubimpact.core.scheduler import (
    PipelineEntry,
    PipelineRegistry,
    scheduler,
)


def test_register_and_schedule_linear():
    """Linear dependency A -> B -> C topological sort."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "A", after=[])
    pr.register_phase("INIT", "B", after=["A"])
    pr.register_phase("INIT", "C", after=["B"])

    order = pr.schedule({"A": [{}], "B": [{}], "C": [{}]}, phase="INIT")
    names = [e.category for e in order]
    assert names == ["A", "B", "C"]


def test_register_and_schedule_parallel():
    """Parallel branches A -> B, A -> C topological sort."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "A", after=[])
    pr.register_phase("INIT", "B", after=["A"])
    pr.register_phase("INIT", "C", after=["A"])

    order = pr.schedule({"A": [{}], "B": [{}], "C": [{}]}, phase="INIT")
    names = [e.category for e in order]
    assert names[0] == "A"
    assert set(names[1:]) == {"B", "C"}


def test_missing_dependency_raises():
    """Dependency on unregistered module should raise ValueError."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "B", after=["NONEXISTENT"])

    with pytest.raises(ValueError, match="NONEXISTENT"):
        pr.schedule({"B": [{}]}, phase="INIT")


def test_cycle_detection():
    """Circular dependency should raise ValueError."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "A", after=["B"])
    pr.register_phase("INIT", "B", after=["A"])

    with pytest.raises(ValueError, match=r"(?i)cycle|circular"):
        pr.schedule({"A": [{}], "B": [{}]}, phase="INIT")


def test_optional_module_not_in_inp_skipped():
    """Optional module not declared in INP is automatically skipped."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "A", after=[])
    pr.register_phase("INIT", "B", after=["A"], required=False)

    order = pr.schedule({"A": [{}]}, phase="INIT")  # B not in config
    names = [e.category for e in order]
    assert names == ["A"]  # B not in INP, skipped


def test_respects_phase():
    """Only modules in the specified phase are scheduled."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "A", after=[])
    pr.register_phase("RUNTIME", "B", after=[])

    order_init = pr.schedule({"A": [{}], "B": [{}]}, phase="INIT")
    assert [e.category for e in order_init] == ["A"]

    order_rt = pr.schedule({"A": [{}], "B": [{}]}, phase="RUNTIME")
    assert [e.category for e in order_rt] == ["B"]


def test_self_loop_raises():
    """Self-referencing dependency should raise ValueError."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "A", after=["A"])

    with pytest.raises(ValueError):
        pr.schedule({"A": [{}]}, phase="INIT")


def test_entry_carries_config():
    """PipelineEntry carries the corresponding INP config dict."""
    pr = PipelineRegistry()
    pr.register_phase("INIT", "A", after=[])

    order = pr.schedule({"A": [{"TYPE": "FOO", "R0": "100.0"}]}, phase="INIT")
    assert order[0].cfg == {"TYPE": "FOO", "R0": "100.0"}
