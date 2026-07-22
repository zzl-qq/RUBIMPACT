"""Tests for OutputCatalog variable definitions."""
import pytest
from rubimpact.infra.output_catalog import output_catalog, OutputCatalog


class TestHistoryVariables:
    def test_u_defined(self):
        entry = output_catalog.get("U")
        assert entry is not None
        assert entry["category"] == "HISTORY"
        assert entry["ndim"] == "n_tip_dof"

    def test_cf_defined(self):
        entry = output_catalog.get("CF")
        assert entry is not None
        assert entry["category"] == "HISTORY"
        assert entry["ndim"] == "n_r"

    def test_pen_defined(self):
        entry = output_catalog.get("PEN")
        assert entry is not None
        assert entry["category"] == "HISTORY"
        assert entry["ndim"] == "n_tip_nodes"

    def test_energy_defined(self):
        entry = output_catalog.get("ENERGY")
        assert entry is not None
        assert entry["category"] == "HISTORY"
        assert entry["ndim"] == 4

    def test_cfn_defined(self):
        entry = output_catalog.get("CFN")
        assert entry is not None
        assert entry["category"] == "HISTORY"
        assert entry["ndim"] == "n_r"
        assert "normal" in entry["description"].lower()

    def test_cft_defined(self):
        entry = output_catalog.get("CFT")
        assert entry is not None
        assert entry["category"] == "HISTORY"
        assert entry["ndim"] == "n_r"
        assert "friction" in entry["description"].lower()


class TestFieldVariables:
    def test_coating_h_defined(self):
        entry = output_catalog.get("COATING_H")
        assert entry is not None
        assert entry["category"] == "FIELD"
        assert entry["ndim"] == "(n_theta,n_x)"

    def test_coating_ep_defined(self):
        entry = output_catalog.get("COATING_EP")
        assert entry is not None
        assert entry["category"] == "FIELD"

    def test_coating_s_defined(self):
        entry = output_catalog.get("COATING_S")
        assert entry is not None
        assert entry["category"] == "FIELD"
        assert entry["ndim"] == "(n_theta,n_x)"

    def test_old_names_removed(self):
        """Old short names H, EP, S must not exist."""
        assert output_catalog.get("H") is None
        assert output_catalog.get("EP") is None
        assert output_catalog.get("S") is None


class TestCatalogBehavior:
    def test_case_insensitive_lookup(self):
        output_catalog.define("testvar", category="HISTORY", source="test", ndim=1)
        assert output_catalog.get("TESTVAR") is not None
        assert output_catalog.get("testvar") is not None

    def test_missing_returns_none(self):
        assert output_catalog.get("NONEXISTENT_VAR_XYZ") is None

    def test_list_names_includes_new_vars(self):
        names = output_catalog.list_names()
        assert "CFN" in names
        assert "CFT" in names
        assert "COATING_H" in names
        assert "COATING_EP" in names
        assert "COATING_S" in names
