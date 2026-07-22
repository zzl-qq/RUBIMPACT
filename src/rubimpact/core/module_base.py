"""Module base class and protocol data classes for the pipeline system."""
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from rubimpact.infra.databus import DataBus


@dataclass
class PipelineStage:
    """A single stage in a pipeline.

    Attributes:
        name: Stage identifier (e.g. "compute_normal_force").
        kernel_ref: Reference to registered kernel as "category/TYPE"
                    (e.g. "constitutive/PLASTIC_COATING_LAW"), or None
                    for inline logic.
        depends_on: List of stage names that must execute before this one.
        optional: If True, stage can be disabled at build time (dead branch elimination).
    """
    name: str
    kernel_ref: str | None
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class PipelineProtocol:
    """Protocol declaration for a module's pipeline contribution.

    Attributes:
        stages: Ordered list of PipelineStage objects the module contributes.
        params: Module-level parameters frozen at build time (numpy arrays, scalars).
    """
    stages: list[PipelineStage]
    params: dict[str, Any] = field(default_factory=dict)


class Module(ABC):
    """Base class for all framework modules.

    Lifecycle:
        1. __init__(db, context) — store DataBus reference and shared context.
        2. configure(cfg) — parse INP config, resolve JIT kernels, build pipeline.
        3. get_pipeline() → callable | None — return the compiled pipeline function.
    """

    def __init__(self, db: DataBus, context: dict[str, Any] | None = None):
        self.db = db
        self.ctx = context or {}

    def configure(self, cfg: dict[str, Any]) -> None:
        """Parse configuration, resolve kernels, build pipeline.

        Override in subclasses. Called once during model assembly.
        """
        pass

    def get_pipeline(self) -> callable | None:
        """Return the compiled pipeline function, or None if no pipeline.

        Override in subclasses that produce a JIT pipeline.
        """
        return None

    def get_pipeline_protocol(self) -> PipelineProtocol | None:
        """Return the pipeline protocol declaration, or None.

        Override in subclasses to declare pipeline stages and parameters.
        Used by PipelineFactory to auto-generate composed pipelines.
        """
        return None
