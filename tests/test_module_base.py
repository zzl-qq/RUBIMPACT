"""Tests for Module base class and protocol data classes."""
import numpy as np
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol


class FakeModule(Module):
    """Minimal module implementation for testing."""
    def configure(self, cfg):
        self.captured_cfg = cfg

    def get_pipeline_protocol(self):
        return PipelineProtocol(
            stages=[
                PipelineStage(
                    name="compute_normal_force",
                    kernel_ref="constitutive/PLASTIC_COATING_LAW",
                    depends_on=[],
                ),
                PipelineStage(
                    name="apply_wear",
                    kernel_ref="wear/PLASTIC_STRAIN",
                    depends_on=["compute_normal_force"],
                    optional=True,
                ),
            ],
            params={"pcl_params": np.array([1.0, 2.0, 3.0, 4.0])},
        )


def test_module_configure():
    """Module.configure() stores context and config."""
    mod = FakeModule(None, {"Omega": 100.0})
    mod.configure({"TYPE": "PCL_CONTACT"})
    assert mod.captured_cfg == {"TYPE": "PCL_CONTACT"}
    assert mod.ctx["Omega"] == 100.0


def test_module_get_pipeline_default():
    """get_pipeline() returns None by default (no pipeline yet)."""
    mod = FakeModule(None, {})
    assert mod.get_pipeline() is None


def test_pipeline_protocol_stages():
    """PipelineProtocol stages are correctly stored."""
    mod = FakeModule(None, {})
    proto = mod.get_pipeline_protocol()
    assert len(proto.stages) == 2
    assert proto.stages[0].name == "compute_normal_force"
    assert proto.stages[0].kernel_ref == "constitutive/PLASTIC_COATING_LAW"
    assert proto.stages[0].depends_on == []
    assert proto.stages[0].optional is False
    assert proto.stages[1].optional is True
    assert "pcl_params" in proto.params


def test_db_accessible():
    """Module can access DataBus."""
    from rubimpact.infra.databus import DataBus
    db = DataBus()
    db.set("test.key", 42)
    mod = FakeModule(db, {})
    assert mod.db.get("test.key") == 42
