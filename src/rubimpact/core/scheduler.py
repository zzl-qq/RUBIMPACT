"""Pipeline scheduler for topological module ordering.

Moved from infra/pipeline_registry.py.  Uses Kahn's algorithm to
topologically sort INIT and RUNTIME phase modules based on their
declared order_after dependencies.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class PipelineEntry:
    """A single entry in the scheduled execution pipeline."""

    category: str  # INP keyword (e.g. "CASING", "TIME_INTEGRATOR")
    phase: str  # "INIT" or "RUNTIME"
    cfg: dict  # INP config entry for this module
    required: bool = True  # Whether the module must be in INP


class PipelineRegistry:
    """Manages execution phases and their topological ordering."""

    def __init__(self):
        # {(phase, category): {"after": [categories], "required": bool}}
        self._phases: dict[tuple[str, str], dict] = {}

    def register_phase(
        self,
        phase: str,
        category: str,
        after: list[str] | None = None,
        required: bool = True,
    ) -> None:
        """Register an execution phase entry.

        Args:
            phase: "INIT" or "RUNTIME"
            category: INP keyword / module category
            after: List of categories that must execute before this one.
            required: If True, INP must declare this keyword.
                      If False, it's optional (e.g. DYNAMIC_RELAXATION, ROM).
        """
        self._phases[(phase.upper(), category.upper())] = {
            "after": [a.upper() for a in (after or [])],
            "required": required,
        }

    def _is_optional_in_any_phase(self, category: str) -> bool:
        """Check if a category is registered as optional in any phase."""
        category = category.upper()
        for (p, c), info in self._phases.items():
            if c == category and not info.get("required", True):
                return True
        return False

    def schedule(self, config: dict, phase: str) -> list[PipelineEntry]:
        """Topological sort of modules in the given phase.

        Args:
            config: Parsed INP config dict (from KeywordParser).
            phase: "INIT" or "RUNTIME"

        Returns:
            Ordered list of PipelineEntry objects.

        Raises:
            ValueError: If a cycle is detected or a dependency is missing.
        """
        phase = phase.upper()
        declared = {k.upper() for k in config if isinstance(config[k], list)}

        # Collect entries for this phase
        entries: dict[str, PipelineEntry] = {}
        for (p, cat), info in self._phases.items():
            if p != phase:
                continue
            if cat in declared:
                kw_entries = config[cat]
                if kw_entries:
                    entries[cat] = PipelineEntry(
                        category=cat,
                        phase=phase,
                        cfg=kw_entries[0],
                        required=info["required"],
                    )
            elif info["required"]:
                raise ValueError(
                    f"Required keyword *{cat} is missing from INP "
                    f"(phase: {phase})"
                )
            # optional + not declared -> skip

        # Build in-degree map
        in_degree: dict[str, int] = {cat: 0 for cat in entries}
        adj: dict[str, list[str]] = {cat: [] for cat in entries}

        for cat, entry in entries.items():
            info = self._phases.get((phase, cat), {})
            for dep in info.get("after", []):
                dep_cat = dep.upper()
                # Self-loop check
                if dep_cat == cat:
                    raise ValueError(f"Circular dependency: {cat} depends on itself")
                # If dependency is a declared keyword, enforce ordering
                if dep_cat in entries:
                    adj.setdefault(dep_cat, []).append(cat)
                    in_degree[cat] = in_degree.get(cat, 0) + 1
                elif dep_cat in declared:
                    # Dependency exists in INP but registered to a different
                    # phase -- that's OK, it will execute before RUNTIME starts.
                    pass
                elif self._is_optional_in_any_phase(dep_cat):
                    # Dependency is registered as optional (required: false)
                    # in some phase but not declared — skip it silently.
                    pass
                else:
                    raise ValueError(
                        f"Module {cat} depends on {dep_cat}, but "
                        f"{dep_cat} is neither declared in INP nor "
                        f"registered in phase '{phase}'"
                    )

        # Kahn's algorithm
        queue = deque([cat for cat, deg in in_degree.items() if deg == 0])
        result: list[PipelineEntry] = []

        while queue:
            current = queue.popleft()
            result.append(entries[current])
            for neighbor in adj.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(entries):
            remaining = set(entries) - {e.category for e in result}
            raise ValueError(
                f"Dependency cycle detected in phase '{phase}'. "
                f"Unresolved: {sorted(remaining)}"
            )

        return result


# Global singleton
scheduler = PipelineRegistry()
