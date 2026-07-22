"""Core infrastructure: registry, pipeline factory, module base, scheduler."""
from rubimpact.core.registry import ComponentRegistry, KernelSpec, ModuleSpec, components
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol
from rubimpact.core.pipeline_factory import PipelineFactory
from rubimpact.core.scheduler import PipelineRegistry, PipelineEntry, scheduler
