"""Module descriptor infrastructure (framework §S10 / future §5.6).

Provides a structured, YAML-based format for describing module interfaces:
provides/requires ports, parameters with defaults and types, and module
metadata.  This is the foundation for future constraint-graph validation,
automatic dispatch, and documentation generation.

Usage::

    from rubimpact.infra.module_descriptor import (
        ModuleDescriptor, PortSpec, ParamSpec, load_descriptor,
    )

    # Programmatic
    desc = ModuleDescriptor(
        id="Stribeck", name="Stribeck Friction", category="friction_force",
        parameters=[ParamSpec(name="mu_s", type="float", default=0.3)],
    )

    # From YAML file
    desc = load_descriptor("modules/friction_force/Stribeck.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
# Port & parameter specs
# ═══════════════════════════════════════════════════════════════

@dataclass
class PortSpec:
    """A single input or output port of a module.

    Attributes:
        name: Port identifier (e.g. ``"contact_force"``).
        type: Data type — ``"scalar"``, ``"vector[n]"``, ``"matrix[m×n]"``,
            ``"dict"``, or ``"any"``.
        dim: Dimension constraint (``None`` = unconstrained).
        description: Human-readable description.
    """
    name: str
    type: str = "any"
    dim: Optional[int] = None
    description: str = ""


@dataclass
class ParamSpec:
    """A single configurable parameter of a module.

    Attributes:
        name: Parameter key (e.g. ``"mu_s"``).
        type: Python type hint string — ``"float"``, ``"int"``, ``"str"``,
            ``"bool"``.
        required: Whether the parameter must be supplied (default ``False``).
        default: Default value when the parameter is omitted.
        description: Human-readable description.
    """
    name: str
    type: str = "float"
    required: bool = False
    default: Any = None
    description: str = ""


# ═══════════════════════════════════════════════════════════════
# Module descriptor
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModuleDescriptor:
    """Structured description of a module's interface and metadata.

    Attributes:
        id: Globally unique module identifier (e.g. ``"Stribeck"``).
        name: Human-readable name (e.g. ``"Stribeck 摩擦模型"``).
        category: Module category / INP keyword (e.g. ``"friction_force"``).
        version: Semantic version string.
        description: Detailed description of the module's purpose and behaviour.
        provides: Ports this module writes to the DataBus.
        requires: Ports this module reads from the DataBus.
        parameters: Configurable parameters accepted via INP.
        lifecycle: Execution lifecycle metadata (phase, required, order_after).
        requires_keywords: INP keywords that must also be declared.
        conflicts_with: Module IDs this module is incompatible with.
        compatible_with: Module IDs this module is known to work with.
        references: Academic or other references.
        tags: Search / classification tags.
    """
    id: str
    name: str = ""
    category: str = ""
    version: str = "1.0"
    description: str = ""
    provides: list[PortSpec] = field(default_factory=list)
    requires: list[PortSpec] = field(default_factory=list)
    parameters: list[ParamSpec] = field(default_factory=list)
    lifecycle: dict = field(default_factory=dict)
    requires_keywords: list[str] = field(default_factory=list)
    compatible_with: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    accepts_children: dict[str, dict] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# YAML loader
# ═══════════════════════════════════════════════════════════════

def _dict_to_port(d: dict) -> PortSpec:
    return PortSpec(
        name=d["name"],
        type=d.get("type", "any"),
        dim=d.get("dim"),
        description=d.get("description", ""),
    )


def _dict_to_param(d: dict) -> ParamSpec:
    return ParamSpec(
        name=d["name"],
        type=d.get("type", "float"),
        required=d.get("required", False),
        default=d.get("default"),
        description=d.get("description", ""),
    )


def load_descriptor(path: str | Path) -> ModuleDescriptor:
    """Load a module descriptor from a YAML file.

    Expected YAML structure::

        module:
          id: "Stribeck"
          name: "Stribeck Friction"
          category: friction_force
          version: "1.0"
          description: "Static/dynamic friction transition model."
          provides:
            - name: friction_force
              type: scalar
              description: "Friction force"
          requires:
            - name: contact_force
              type: scalar
            - name: sliding_velocity
              type: scalar
          parameters:
            - name: mu_s
              type: float
              default: 0.3
              description: "Static friction coefficient"
            - name: mu_d
              type: float
              default: 0.25
              description: "Dynamic friction coefficient"
            - name: v_s
              type: float
              default: 1.0
              description: "Stribeck characteristic velocity"
          compatible_with: []
          conflicts_with: []
          references: []
          tags: ["friction", "stribeck"]
    """
    import yaml

    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    m = raw.get("module", raw)
    return ModuleDescriptor(
        id=m["id"],
        name=m.get("name", m["id"]),
        category=m.get("category", ""),
        version=m.get("version", "1.0"),
        description=m.get("description", ""),
        provides=[_dict_to_port(p) for p in m.get("provides", [])],
        requires=[_dict_to_port(r) for r in m.get("requires", [])],
        parameters=[_dict_to_param(p) for p in m.get("parameters", [])],
        lifecycle=m.get("lifecycle", {}),
        requires_keywords=m.get("requires_keywords", []),
        compatible_with=m.get("compatible_with", []),
        conflicts_with=m.get("conflicts_with", []),
        accepts_children=m.get("accepts_children", {}),
        references=m.get("references", []),
        tags=m.get("tags", []),
    )


# ═══════════════════════════════════════════════════════════════
# Auto-discovery & validation
# ═══════════════════════════════════════════════════════════════

def discover_modules(modules_dir: str | Path = "modules",
                     use_cache: bool = True) -> dict[str, ModuleDescriptor]:
    """Auto-discover YAML module descriptors in a directory tree.

    Scans ``modules_dir`` recursively for ``*.yaml`` files, loads each,
    and returns a dict keyed by ``"CATEGORY/ID"`` (e.g. ``"CASING/LOBE"``).

    Results are cached by directory mtime — subsequent calls with the same
    ``modules_dir`` return instantly unless a YAML file has been modified.

    Returns an empty dict if the directory does not exist or contains
    no valid descriptors.
    """
    import logging
    logger = logging.getLogger(__name__)

    root = Path(modules_dir)
    if not root.is_dir():
        logger.debug("Module descriptor directory not found: %s", root)
        return {}

    # ── mtime-based cache ────────────────────────────────────────
    if use_cache:
        latest_mtime = 0.0
        yaml_files = []
        try:
            for yp in root.rglob("*.yaml"):
                yaml_files.append(yp)
                mtime = yp.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
        except OSError:
            latest_mtime = 0.0

        cache_key = str(root.resolve())
        cached = _descriptor_cache.get(cache_key)
        if cached is not None and cached[0] >= latest_mtime:
            logger.debug("Using cached module descriptors (%d files, "
                         "mtime=%.3f)", len(cached[1]), cached[0])
            return cached[1]

    # ── (Re-)discover ────────────────────────────────────────────
    discovered: dict[str, ModuleDescriptor] = {}
    yaml_paths = sorted(root.rglob("*.yaml")) if not (use_cache and yaml_files) else yaml_files
    for yaml_path in sorted(yaml_paths):
        try:
            desc = load_descriptor(yaml_path)
            # Infer category from path if empty
            if not desc.category:
                desc.category = _infer_category_from_path(str(yaml_path))
            key = f"{desc.category.upper()}/{desc.id.upper()}"
            discovered[key] = desc
            logger.debug("Discovered module descriptor: %s", key)
        except Exception as exc:
            logger.warning("Failed to load module descriptor %s: %s",
                          yaml_path, exc)

    # ── Update cache ─────────────────────────────────────────────
    if use_cache:
        cache_key = str(root.resolve())
        _descriptor_cache[cache_key] = (latest_mtime, discovered)

    return discovered


# Module-level cache for discover_modules()
_descriptor_cache: dict[str, tuple[float, dict[str, ModuleDescriptor]]] = {}


def validate_children(descriptor: ModuleDescriptor,
                      declared_submodules: dict) -> list[str]:
    """Validate INP-declared submodules against a descriptor's
    ``accepts_children`` specification.

    Args:
        descriptor: The parent module descriptor (e.g. CONSTITUTIVE/
            PLASTIC_COATING_LAW).
        declared_submodules: Submodules from INP, keyed by slot name,
            each a dict with at least a ``TYPE`` key, e.g.
            ``{"hardening": {"TYPE": "LINEAR_ISOTROPIC", "K_plas": "1000.0"}}``.

    Returns:
        List of error strings (empty = all submodules valid).
    """
    errors: list[str] = []
    for child_slot, child_spec in descriptor.accepts_children.items():
        candidates: list[str] = child_spec.get("candidates", [])
        cardinality: int = child_spec.get("cardinality", 1)
        declared = declared_submodules.get(child_slot)
        if declared is None:
            has_default = "default" in child_spec
            if cardinality > 0 and not has_default:
                errors.append(
                    f"{descriptor.category}/{descriptor.id}: "
                    f"requires submodule '{child_slot}' "
                    f"(accepted: {candidates})"
                )
            continue
        declared_type = declared.get("TYPE", "")
        if declared_type and declared_type.upper() not in [c.upper() for c in candidates]:
            errors.append(
                f"{descriptor.category}/{descriptor.id}: "
                f"submodule '{child_slot}' TYPE={declared_type} "
                f"not in accepted candidates: {candidates}"
            )
    return errors


def validate_config(discovered: dict[str, ModuleDescriptor],
                    config: dict) -> list[str]:
    """Validate the full INP config against all discovered descriptors.

    Checks:
    1. **Required keywords**: if a descriptor's ``lifecycle.required`` is
       true, the INP must contain that keyword.
    2. **Cross-module constraints**: if a descriptor declares
       ``requires_keywords``, those keywords must also be present in the
       INP (and if the current module is declared).  If a descriptor
       declares ``conflicts_with``, the INP must NOT contain conflicting
       module IDs.
    3. **Submodule compatibility**: each module's declared submodules are
       validated against its ``accepts_children`` spec (TYPE must be in
       the candidates list, required submodules must be present).

    The mapping from INP keyword to descriptor category is derived
    dynamically from the discovered descriptors — no hardcoded mapping.
    New categories are picked up automatically when a YAML descriptor
    is added for them.

    Returns a list of validation error strings (empty = valid).
    """
    errors: list[str] = []
    # Build a per-category lookup: {INP_KEYWORD: [descriptor, ...]}
    by_category: dict[str, list[ModuleDescriptor]] = {}
    for desc in discovered.values():
        cat = desc.category.upper()
        by_category.setdefault(cat, []).append(desc)

    # ── 1. Required-keyword check ──────────────────────────────
    declared_keywords = {k.upper() for k in config if k != "MODEL"}
    missing_required: set[str] = set()
    for desc in discovered.values():
        lifecycle = desc.lifecycle or {}
        if lifecycle.get("required", False):
            cat = desc.category.upper()
            if cat not in declared_keywords:
                missing_required.add(cat)
    for cat in sorted(missing_required):
        errors.append(f"Required keyword *{cat} is missing from INP")

    # ── 2. Cross-module constraint checks ──────────────────────
    # Collect all declared module TYPEs: {CATEGORY: TYPE}
    declared_types: dict[str, str] = {}
    for kw, entries in config.items():
        if isinstance(entries, list) and entries:
            etype = entries[0].get("TYPE", "")
            if etype:
                declared_types[kw.upper()] = etype.upper()

    for cat, descs in by_category.items():
        entries = config.get(cat, [])
        if not entries:
            continue
        entry = entries[0]
        etype = entry.get("TYPE", "")

        if etype:
            key = f"{cat}/{etype.upper()}"
        else:
            # Category-level keyword without TYPE variant
            # (e.g. *MATRIX_ASSEMBLY has no TYPE but needs submodule validation)
            key = f"{cat}/{cat}"
        desc = discovered.get(key)
        if desc is None:
            continue

        # 2a. requires_keywords: declared module requires these keywords
        for req_kw in (desc.requires_keywords or []):
            if req_kw.upper() not in declared_keywords:
                errors.append(
                    f"{key}: requires keyword *{req_kw} but it is "
                    f"not declared in INP")

        # 2b. conflicts_with: declared module incompatible with these IDs
        for conflict_id in (desc.conflicts_with or []):
            # conflict_id format: "CATEGORY/ID" or just "ID"
            if "/" in conflict_id:
                c_cat, c_id = conflict_id.split("/", 1)
                if (c_cat.upper() in declared_types
                        and declared_types[c_cat.upper()] == c_id.upper()):
                    errors.append(
                        f"{key} conflicts with {conflict_id} "
                        f"(declared in INP)")
            # TODO: plain-ID conflict resolution (search across
            #       all declared types)

        # 2c. Submodule validation
        submodules = entry.get("submodules", {})
        child_errors = validate_children(desc, submodules)
        errors.extend(child_errors)

    # ── 4. Port compatibility check ──────────────────────────
    port_errors = validate_ports(discovered)
    errors.extend(port_errors)
    return errors


# ═══════════════════════════════════════════════════════════════
# Port validation & path inference
# ═══════════════════════════════════════════════════════════════


def _infer_category_from_path(yaml_path: str) -> str:
    """Infer module category from directory path.

    ``modules/CONSTITUTIVE/hardening/LINEAR_ISOTROPIC.yaml``
    -> ``CONSTITUTIVE/hardening``

    The first directory segment is the INP keyword category.
    Additional segments are submodule slot names.
    """
    parts = Path(yaml_path).parts
    # Find 'modules' segment and take everything after it except the filename
    try:
        idx = parts.index("modules")
    except ValueError:
        return ""
    # parts = ["...", "modules", "CONSTITUTIVE", "hardening", "LINEAR_ISOTROPIC.yaml"]
    category_parts = parts[idx + 1:-1]  # skip filename
    if not category_parts:
        return ""
    return "/".join(category_parts)


def _types_compatible(provides_type: str, requires_type: str) -> bool:
    """Check if two port types are compatible.

    Rules:
    - "any" on either side -> compatible
    - Exact match -> compatible
    - "ndarray" <- "vector[n]" (vector is ndarray with dim)
    - "scalar" -> "scalar" only
    - "matrix" -> "matrix" only
    - "dict" -> "dict" only
    - "list" -> "list" only
    """
    if provides_type == "any" or requires_type == "any":
        return True
    if provides_type == requires_type:
        return True
    # ndarray can satisfy typed vector/matrix (checked by dim separately)
    if provides_type == "ndarray" and requires_type.startswith("vector"):
        return True
    if provides_type == "ndarray" and requires_type.startswith("matrix"):
        return True
    return False


def validate_ports(discovered: dict[str, "ModuleDescriptor"]) -> list[str]:
    """Validate port compatibility across lifecycle-bearing modules.

    Only modules that declare a ``lifecycle`` (INIT or RUNTIME phase) are
    checked — these are the top-level INP keyword modules that communicate
    via DataBus.  Submodule descriptors (damping, friction, hardening,
    JIT kernels, …) without a lifecycle are internal implementation
    details and do NOT participate in DataBus-level port exchange.

    For each module's ``requires`` ports, check that:
    1. Some lifecycle module in ``discovered`` ``provides`` that port name.
    2. The port types are compatible (scalar -> scalar, vector -> vector,
       matrix -> matrix, ndarray -> ndarray, any -> anything).

    ``PortSpec.type == "any"`` means unconstrained — compatible with
    everything.  ``dim`` mismatches are also checked.

    Returns a list of error strings (empty = all ports compatible).
    """
    errors: list[str] = []

    # Only lifecycle-bearing modules participate in DataBus port exchange
    lifecycle_modules = {
        key: desc for key, desc in discovered.items()
        if desc.lifecycle
    }

    # Build provider index: port_name -> [(provider_key, PortSpec), ...]
    providers: dict[str, list[tuple[str, "PortSpec"]]] = {}
    for key, desc in lifecycle_modules.items():
        for port in (desc.provides or []):
            providers.setdefault(port.name, []).append((key, port))

    # Check each consumer's requires
    for key, desc in lifecycle_modules.items():
        for port in (desc.requires or []):
            if port.name not in providers:
                errors.append(
                    f"{key}: requires port '{port.name}' "
                    f"but no module provides it"
                )
                continue
            for prov_key, prov_port in providers[port.name]:
                if not _types_compatible(prov_port.type, port.type):
                    errors.append(
                        f"{key}: port '{port.name}' requires type "
                        f"'{port.type}' but {prov_key} provides "
                        f"'{prov_port.type}'"
                    )
                if prov_port.dim is not None and port.dim is not None:
                    if prov_port.dim != port.dim:
                        errors.append(
                            f"{key}: port '{port.name}' requires dim "
                            f"{port.dim} but {prov_key} provides "
                            f"dim {prov_port.dim}"
                        )
    return errors
