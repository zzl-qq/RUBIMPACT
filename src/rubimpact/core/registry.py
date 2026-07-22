"""Unified ComponentRegistry: single namespace for KernelSpec and ModuleSpec."""
from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class KernelSpec:
    """L3 scalar JIT kernel descriptor."""
    fn: Callable[..., Any]   # @njit decorated function
    signature: str            # function signature identifier (compile-time check)
    stage: str                # role in the pipeline


@dataclass
class ModuleSpec:
    """L2/L1 instantiable module descriptor."""
    builder: Callable[..., Any]  # (db: DataBus, ctx: dict) -> Module instance
    protocol: str                 # protocol name (e.g. "NormalForceProtocol")
    required: bool = False        # whether INP must declare this module



class ComponentRegistry:
    """Unified registry for all components (kernels + modules).

    Single (category, type_name) namespace replaces both
    ModuleRegistry and JITRegistry.
    """

    def __init__(self):
        self._components: dict[tuple[str, str], KernelSpec | ModuleSpec] = {}
        self._orchestrators: dict[str, list[str]] = {}
        self._classes: dict[str, type] = {}
        self._typed_classes: dict[tuple[str, str], type] = {}

    # ── Core registration ──

    def register(self, category: str, type_name: str,
                 component: KernelSpec | ModuleSpec | callable) -> None:
        """Register a component under (category, type_name).

        Backward compat: if component is a bare callable (not KernelSpec or
        ModuleSpec), wraps it as ModuleSpec(builder=component).

        Args:
            category: Component category (e.g. "constitutive", "contact_force").
            type_name: TYPE value (e.g. "PLASTIC_COATING_LAW", "PCL_CONTACT").
            component: KernelSpec, ModuleSpec, or a bare callable (builder).
        """
        if not isinstance(component, (KernelSpec, ModuleSpec)):
            # Backward compat: bare callable → ModuleSpec
            component = ModuleSpec(builder=component, protocol="__unknown__")
        key = (category.upper(), type_name.upper())
        self._components[key] = component

    def resolve(self, category: str, type_name: str) -> KernelSpec | ModuleSpec | None:
        """Look up a component. Returns None if not registered.

        Lookup is case-insensitive for both category and type_name.
        """
        key = (category.upper(), type_name.upper())
        return self._components.get(key)

    def resolve_kernel(self, category: str, type_name: str) -> KernelSpec | None:
        """Look up a KernelSpec. Returns None if not found or wrong type."""
        result = self.resolve(category, type_name)
        if isinstance(result, KernelSpec):
            return result
        return None

    def resolve_module(self, category: str, type_name: str) -> ModuleSpec | None:
        """Look up a ModuleSpec. Returns None if not found or wrong type."""
        result = self.resolve(category, type_name)
        if isinstance(result, ModuleSpec):
            return result
        return None

    def list_types(self, category: str) -> list[str]:
        """List all registered TYPE names for a category."""
        cat = category.upper()
        return sorted([
            tn for (c, tn) in self._components if c == cat
        ])

    def categories(self) -> list[str]:
        """List all registered categories."""
        return sorted(set(c for (c, _) in self._components))

    # ── Backward compat: returns kernel function directly (JITRegistry API) ──

    def get(self, category: str, type_name: str):
        """Backward compat with JITRegistry + ModuleRegistry.

        For KernelSpec: returns .fn (the @njit function).
        For ModuleSpec: returns .builder (the builder callable).
        Returns None if not found.
        """
        spec = self.resolve(category, type_name)
        if isinstance(spec, KernelSpec):
            return spec.fn
        if isinstance(spec, ModuleSpec):
            return spec.builder
        return None

    # ── Class registration (for direct module instantiation) ──

    def register_typed_class(self, category: str, type_name: str, cls: type) -> None:
        """Register a class for a specific (category, type_name). Backward compat."""
        self._typed_classes[(category.upper(), type_name.upper())] = cls

    def get_typed_class(self, category: str, type_name: str) -> type | None:
        """Look up a typed class. Falls back to get_class()."""
        return self._typed_classes.get(
            (category.upper(), type_name.upper()),
            self._classes.get(category.upper()))

    def register_class(self, category: str, cls: type) -> None:
        """Register a module class for direct instantiation by category name.

        Args:
            category: Module category name (e.g. "TIME_INTEGRATOR").
            cls: Module class (must be a subclass of Module).
        """
        self._classes[category.upper()] = cls

    def resolve_class(self, category: str) -> type | None:
        """Look up a registered module class. Returns None if not found."""
        return self._classes.get(category.upper())

    # ── Orchestrator registration (for auto test-matrix generation) ──

    def register_orchestrator(self, name: str, slots: list[str]) -> None:
        """Register an orchestrator's slot names for combination enumeration.

        Args:
            name: Orchestrator name (e.g. "ForceAssembler").
            slots: Ordered list of category slot names that accept TYPE variants.
        """
        self._orchestrators[name.upper()] = (list(slots), [s.upper() for s in slots])

    def list_combinations(self, orchestrator: str) -> list[dict[str, str]]:
        """Generate all valid TYPE combinations for an orchestrator.

        Returns a list of dicts mapping slot_name → type_name for each
        combination. Used to auto-generate L2 pipeline comparison tests.

        Example:
            [{"contact_force": "PCL_CONTACT", "friction_force": "COULOMB"},
             {"contact_force": "PCL_CONTACT", "friction_force": "STRIBECK"}, ...]
        """
        name = orchestrator.upper()
        entry = self._orchestrators.get(name)
        if not entry:
            return []
        original_slots, lookup_slots = entry

        # Collect available types per slot
        type_lists = []
        for slot in lookup_slots:
            types_for_slot = []
            for (c, tn), comp in self._components.items():
                if c == slot and isinstance(comp, ModuleSpec):
                    types_for_slot.append(tn)
            if not types_for_slot:
                return []
            type_lists.append(types_for_slot)

        # Cartesian product
        result = []
        for combo in itertools.product(*type_lists):
            result.append(dict(zip(original_slots, combo)))
        return result


# Global singleton
components = ComponentRegistry()
