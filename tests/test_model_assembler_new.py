"""Smoke tests for new ModelAssembler — zero special branches.

Verifies:
  - KeywordParser parses minimal INP
  - ModelAssembler instantiates
  - All module classes registered in components
  - All module classes support configure()
  - Runtime modules support get_pipeline()
  - DRModule instantiates and configures
"""

import pytest


MINIMAL_INP = """*MODEL, NAME=SMOKE
*CASING, TYPE=CYLINDRICAL
    *AXIAL_SHAPE, TYPE=CYLINDRICAL
    *CIRCUMFERENTIAL_SHAPE, TYPE=UNIFORM
R0=1.0
*COATING, TYPE=UNIFORM_GRID
h_coat=0.01, L=0.1, n_theta=36, n_x=10
*EXTERNAL_DATA
    *MATRIX, ROLE=MASS, FILE=dummy.mtx
    *MATRIX, ROLE=STIFFNESS, FILE=dummy.mtx, Omega=0.0
    *NODES, ROLE=TIP, FILE=dummy.csv
*MATRIX_ASSEMBLY
    *MASS, TYPE=ORIGINAL
    *STIFFNESS, TYPE=DIRECT
    *DAMPING, TYPE=NONE
*STEP
Omega=314.0, h=1e-6, T_f=1e-4
*OUTPUT, TYPE=HISTORY, FREQUENCY=1
U, CF
*OUTPUT, TYPE=FIELD, FREQUENCY=100
COATING_EP
*END STEP
*END MODEL
"""


class TestKeywordParser:
    """KeywordParser is unchanged from original model_assembler.py."""

    def test_parse_minimal(self):
        from rubimpact.orchestrators.model_assembler import KeywordParser
        config = KeywordParser().parse(MINIMAL_INP)
        assert "MODEL" in config
        assert config["MODEL"]["NAME"] == "SMOKE"
        assert "STEP" in config
        assert len(config["STEP"]) == 1
        step = config["STEP"][0]
        assert step["params"]["Omega"] == "314.0"
        assert step["params"]["h"] == "1e-6"
        assert step["params"]["T_f"] == "1e-4"
        assert len(step["outputs"]) == 2

    def test_parse_params(self):
        from rubimpact.orchestrators.model_assembler import KeywordParser
        result = KeywordParser._parse_params("a=1, b=2.0, c=hello")
        assert result == {"a": "1", "b": "2.0", "c": "hello"}

    def test_parse_keyword_line(self):
        from rubimpact.orchestrators.model_assembler import KeywordParser
        kw, params = KeywordParser()._parse_keyword_line(
            "*CASING, TYPE=CYLINDRICAL")
        assert kw == "CASING"
        assert params == {"TYPE": "CYLINDRICAL"}

    def test_top_level_keywords(self):
        from rubimpact.orchestrators.model_assembler import KeywordParser
        config = KeywordParser().parse(MINIMAL_INP)
        assert "CASING" in config
        assert "COATING" in config
        assert "EXTERNAL_DATA" in config
        assert "MATRIX_ASSEMBLY" in config
        assert len(config["CASING"]) == 1
        assert config["CASING"][0]["R0"] == "1.0"

    def test_submodules_parsed(self):
        from rubimpact.orchestrators.model_assembler import KeywordParser
        config = KeywordParser().parse(MINIMAL_INP)
        ed = config["EXTERNAL_DATA"][0]
        assert len(ed["sub_list"]) == 3  # 2 matrices + 1 nodes
        ma = config["MATRIX_ASSEMBLY"][0]
        assert "mass" in ma["submodules"]
        assert "stiffness" in ma["submodules"]
        assert "damping" in ma["submodules"]


class TestModelAssemblerInit:
    """ModelAssembler instantiation and registration checks."""

    def test_instantiate(self):
        from rubimpact.orchestrators.model_assembler import ModelAssembler
        ma = ModelAssembler()
        assert ma.db is not None
        assert ma.job is not None

    def test_components_registered(self):
        """All module classes in components registry."""
        from rubimpact.core.registry import components
        for cat in ["EXTERNAL_DATA", "CASING", "COATING", "MATRIX_ASSEMBLY",
                     "ROM", "CONTACT_DETECTOR", "CONSTITUTIVE",
                     "TIME_INTEGRATOR", "FORCE_ASSEMBLER",
                     "DYNAMIC_RELAXATION"]:
            cls = components.resolve_class(cat)
            assert cls is not None, f"{cat} not registered in components"

    def test_init_modules_have_configure(self):
        """Init module classes support configure() after _adapt_modules()."""
        from rubimpact.init.external_data import ExternalData
        from rubimpact.init.casing import Casing
        from rubimpact.init.coating import Coating
        from rubimpact.init.matrix_assembly import MatrixAssembly
        from rubimpact.init.rom import ROM

        for cls in [ExternalData, Casing, Coating, MatrixAssembly, ROM]:
            assert hasattr(cls, "configure"), \
                f"{cls.__name__} missing configure"
            assert callable(cls.configure), \
                f"{cls.__name__}.configure not callable"

    def test_runtime_modules_have_configure(self):
        """Runtime module classes support configure()."""
        from rubimpact.modules.contact_detector import ContactDetector
        from rubimpact.runtime.constitutive import ConstitutiveModule
        from rubimpact.modules.time_integrator import TimeIntegrator
        from rubimpact.orchestrators.force_assembler import ForceAssembler

        for cls in [ContactDetector, TimeIntegrator, ForceAssembler]:
            assert hasattr(cls, "configure"), \
                f"{cls.__name__} missing configure"
            assert callable(cls.configure), \
                f"{cls.__name__}.configure not callable"
        # ConstitutiveModule gets configure via monkey-patch in model_assembler
        assert hasattr(ConstitutiveModule, "configure"), \
            "ConstitutiveModule missing configure"

    def test_runtime_have_get_pipeline(self):
        """ContactDetector and ForceAssembler have get_pipeline()."""
        from rubimpact.modules.contact_detector import ContactDetector
        from rubimpact.orchestrators.force_assembler import ForceAssembler
        from rubimpact.modules.time_integrator import TimeIntegrator

        for cls in [ContactDetector, ForceAssembler, TimeIntegrator]:
            assert hasattr(cls, "get_pipeline"), \
                f"{cls.__name__} missing get_pipeline"
            assert callable(cls.get_pipeline), \
                f"{cls.__name__}.get_pipeline not callable"

    def test_modules_accept_context(self):
        """Module __init__ now accepts (db, context=None)."""
        from rubimpact.infra.databus import DataBus
        db = DataBus()
        from rubimpact.init.casing import Casing
        from rubimpact.modules.contact_detector import ContactDetector

        # Should not raise: Casing.__init__ was patched to accept context
        c = Casing(db, {"inp_dir": "/tmp", "Omega": 314.0})
        assert c.ctx["Omega"] == 314.0

        cd = ContactDetector(db, context={"h": 1e-6})
        assert cd.ctx["h"] == 1e-6


class TestDRModule:
    """DynamicRelaxation replaces standalone dynamic_relaxation()."""

    def test_instantiate_and_configure(self):
        from rubimpact.modules.dynamic_relaxation import DynamicRelaxation
        dr = DynamicRelaxation(None, {"test": 1})
        assert dr.ctx["test"] == 1
        dr.configure({"max_steps": "100", "tolerance": "1e-8",
                      "relaxation": "0.5", "force_tol": "1e-6"})
        assert dr.max_steps == 100
        assert dr.tol == 1e-8
        assert dr.get_pipeline() is None

    def test_run_no_contact(self):
        """DR with mock detect/assemble — no contact, should converge immediately."""
        import numpy as np
        from rubimpact.modules.dynamic_relaxation import DynamicRelaxation

        dr = DynamicRelaxation(None)
        dr.configure({"max_steps": "100", "tolerance": "1e-10",
                      "relaxation": "0.5", "force_tol": "1e-6"})

        n_r = 3
        K_r = np.eye(n_r, dtype=np.float64)

        def mock_detect(u, t, Omega):
            return (np.zeros(0, dtype=np.float64),
                    np.zeros((0, 5), dtype=np.float64),
                    np.zeros((0, 9), dtype=np.float64))

        def mock_assemble(pen, coords, interp, Omega):
            return np.zeros(n_r, dtype=np.float64)

        u, converged, stats = dr.run(
            mock_detect, mock_assemble,
            np.zeros(n_r), K_r, 314.0)
        assert converged
        assert u is not None
        assert stats["steps"] <= 30  # no contact = fast convergence


class TestSchedulerPhases:
    """Scheduler has all phases registered."""

    def test_init_phases_registered(self):
        from rubimpact.core.scheduler import scheduler
        from rubimpact.orchestrators.model_assembler import KeywordParser

        config = KeywordParser().parse(MINIMAL_INP)
        entries = scheduler.schedule(config, phase="INIT")
        categories = {e.category for e in entries}
        expected = {"EXTERNAL_DATA", "CASING", "COATING", "MATRIX_ASSEMBLY"}
        for cat in expected:
            assert cat in categories, f"INIT phase missing {cat}"

    def test_rom_optional(self):
        """ROM is not required — schedule should not fail without it."""
        from rubimpact.core.scheduler import scheduler
        from rubimpact.orchestrators.model_assembler import KeywordParser

        config = KeywordParser().parse(MINIMAL_INP)
        entries = scheduler.schedule(config, phase="INIT")
        categories = {e.category for e in entries}
        # ROM is optional, should not appear when not declared
        assert "ROM" not in categories
