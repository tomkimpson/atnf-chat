"""Tests for LLM tools module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from atnf_chat.tools import (
    query_catalogue,
    get_pulsar_info,
    generate_query_plan,
    compute_derived_parameter,
    statistical_analysis,
    correlation_analysis,
    get_tools_for_claude,
    get_tool_names,
)
from atnf_chat.tools.query import QueryResult
from atnf_chat.tools.derived import DerivedParameterResult
from atnf_chat.tools.analysis import StatisticalSummary, CorrelationResult


class TestQueryCatalogue:
    """Tests for query_catalogue function."""

    @pytest.fixture
    def mock_catalogue(self) -> MagicMock:
        """Create a mock catalogue interface."""
        mock = MagicMock()
        mock.version = "2.0.0"

        # Mock query results
        df = pd.DataFrame({
            "JNAME": ["J0030+0451", "J0437-4715", "J1012+5307"],
            "P0": [0.00487, 0.00576, 0.00525],
            "DM": [4.33, 2.64, 9.02],
        })

        # Mock provenance
        provenance = MagicMock()
        provenance.catalogue_version = "2.0.0"
        provenance.result_count = 3
        provenance.null_counts = {"JNAME": 0, "P0": 0, "DM": 0}
        provenance.completeness = {"JNAME": 1.0, "P0": 1.0, "DM": 1.0}
        provenance.to_dict.return_value = {}

        mock.query.return_value = (df, provenance)
        return mock

    def test_query_valid_dsl(self, mock_catalogue: MagicMock) -> None:
        """Test query with valid DSL."""
        query_dsl = {
            "select_fields": ["JNAME", "P0"],
            "filters": {
                "op": "and",
                "clauses": [{"field": "P0", "cmp": "lt", "value": 0.03}],
            },
            "limit": 100,
        }

        result = query_catalogue(query_dsl, catalogue=mock_catalogue)

        assert result.success
        assert result.data is not None
        assert len(result.data) == 3
        assert result.error is None

    def test_query_invalid_field(self) -> None:
        """Test query with invalid field."""
        query_dsl = {
            "select_fields": ["INVALID_FIELD"],
        }

        result = query_catalogue(query_dsl)

        assert not result.success
        assert "validation failed" in result.error.lower()
        assert len(result.suggestions) > 0

    def test_query_missing_value(self) -> None:
        """Test query with missing required value."""
        query_dsl = {
            "filters": {
                "op": "and",
                "clauses": [{"field": "P0", "cmp": "lt"}],  # Missing value
            },
        }

        result = query_catalogue(query_dsl)

        assert not result.success
        assert "requires a value" in result.error.lower()

    def test_query_result_format_for_display(self, mock_catalogue: MagicMock) -> None:
        """Test formatting query result for display."""
        query_dsl = {"select_fields": ["JNAME", "P0"]}
        result = query_catalogue(query_dsl, catalogue=mock_catalogue)

        display = result.format_for_display()
        assert "Found" in display or "JNAME" in display

    def test_query_result_to_dict(self, mock_catalogue: MagicMock) -> None:
        """Test converting query result to dictionary."""
        query_dsl = {"select_fields": ["JNAME"]}
        result = query_catalogue(query_dsl, catalogue=mock_catalogue)

        d = result.to_dict()
        assert d["success"] is True
        assert "result_count" in d
        assert "preview" in d


class TestGetPulsarInfo:
    """Tests for get_pulsar_info function."""

    @pytest.fixture
    def mock_catalogue(self) -> MagicMock:
        """Create mock catalogue with get_pulsar support."""
        mock = MagicMock()

        # Mock pulsar lookup
        pulsar_data = pd.Series({
            "JNAME": "J0437-4715",
            "P0": 0.00576,
            "DM": 2.64,
            "BSURF": 2.9e8,
        })
        mock.get_pulsar.return_value = pulsar_data

        return mock

    def test_get_existing_pulsar(self, mock_catalogue: MagicMock) -> None:
        """Test getting an existing pulsar."""
        result = get_pulsar_info("J0437-4715", catalogue=mock_catalogue)

        assert result.success
        assert result.data is not None
        assert len(result.data) == 1

    def test_get_nonexistent_pulsar(self) -> None:
        """Test getting a non-existent pulsar."""
        with patch("atnf_chat.tools.query.get_catalogue") as mock_get:
            mock_cat = MagicMock()
            mock_cat.get_pulsar.return_value = None
            mock_cat.search_pulsars.return_value = pd.DataFrame()
            mock_get.return_value = mock_cat

            result = get_pulsar_info("J9999+9999")

            assert not result.success
            assert "not found" in result.error.lower()


class TestGenerateQueryPlan:
    """Tests for generate_query_plan function."""

    def test_generate_plan_simple(self) -> None:
        """Test generating plan for simple query."""
        query_dsl = {
            "select_fields": ["JNAME", "P0"],
            "filters": {
                "op": "and",
                "clauses": [{"field": "P0", "cmp": "lt", "value": 0.03}],
            },
        }

        result = generate_query_plan(query_dsl)

        assert result["success"]
        assert "plan" in result
        assert "python_code" in result
        assert "QueryATNF" in result["python_code"]

    def test_generate_plan_invalid_query(self) -> None:
        """Test generating plan for invalid query."""
        query_dsl = {
            "select_fields": ["INVALID"],
        }

        result = generate_query_plan(query_dsl)

        assert not result["success"]
        assert "error" in result


class TestComputeDerivedParameter:
    """Tests for compute_derived_parameter function."""

    @pytest.fixture
    def pulsar_df(self) -> pd.DataFrame:
        """Create sample pulsar DataFrame."""
        return pd.DataFrame({
            "JNAME": ["J0030+0451", "J0437-4715", "J1012+5307"],
            "P0": [0.00487, 0.00576, 0.00525],
            "P1": [1.0e-20, 5.7e-20, 1.7e-20],
            "F0": [205.5, 173.7, 190.5],
            "F1": [-4.2e-16, -1.7e-15, -6.2e-16],
            "BSURF": [1.5e8, 2.9e8, 1.9e8],  # ATNF-native
            "PMRA": [5.0, 121.4, 2.6],
            "PMDEC": [-3.0, -71.4, -25.3],
            "DIST": [0.3, 0.16, 0.52],
        })

    def test_compute_bsurf_atnf_native(self, pulsar_df: pd.DataFrame) -> None:
        """Test computing BSURF uses ATNF-native when available."""
        result = compute_derived_parameter(pulsar_df, "BSURF")

        assert result.source == "atnf_native"
        assert result.parameter == "BSURF"
        assert len(result.values) == 3

    def test_compute_bsurf_computed(self, pulsar_df: pd.DataFrame) -> None:
        """Test computing BSURF when not in DataFrame."""
        df_no_bsurf = pulsar_df.drop(columns=["BSURF"])
        result = compute_derived_parameter(df_no_bsurf, "BSURF")

        assert result.source == "computed"
        assert "sqrt" in result.formula.lower()
        assert "braking_mechanism" in result.assumptions

    def test_compute_edot(self, pulsar_df: pd.DataFrame) -> None:
        """Test computing EDOT."""
        df = pulsar_df.drop(columns=["BSURF"])  # Force computation
        result = compute_derived_parameter(df, "EDOT", use_atnf_native=False)

        assert result.source == "computed"
        assert result.parameter == "EDOT"
        assert "moment_of_inertia" in result.assumptions

    def test_compute_age(self, pulsar_df: pd.DataFrame) -> None:
        """Test computing characteristic age."""
        result = compute_derived_parameter(pulsar_df, "AGE", use_atnf_native=False)

        assert result.source == "computed"
        assert "braking_index" in result.assumptions
        # Ages should be positive and reasonable
        assert all(result.values > 0)

    def test_compute_vtrans(self, pulsar_df: pd.DataFrame) -> None:
        """Test computing transverse velocity."""
        result = compute_derived_parameter(pulsar_df, "VTRANS", use_atnf_native=False)

        assert result.source == "computed"
        assert result.parameter == "VTRANS"

    def test_compute_unknown_parameter(self, pulsar_df: pd.DataFrame) -> None:
        """Test computing unknown parameter raises error."""
        with pytest.raises(ValueError, match="Unknown derived parameter"):
            compute_derived_parameter(pulsar_df, "UNKNOWN")


class TestStatisticalAnalysis:
    """Tests for statistical_analysis function."""

    @pytest.fixture
    def pulsar_df(self) -> pd.DataFrame:
        """Create sample pulsar DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "P0": np.random.lognormal(-5, 1, 100),
            "DM": np.random.exponential(50, 100),
            "BSURF": np.random.lognormal(20, 2, 100),
        })

    def test_statistical_analysis_all(self, pulsar_df: pd.DataFrame) -> None:
        """Test statistical analysis on all columns."""
        result = statistical_analysis(pulsar_df)

        assert result.success
        assert len(result.summaries) == 3

    def test_statistical_analysis_specific(self, pulsar_df: pd.DataFrame) -> None:
        """Test statistical analysis on specific parameters."""
        result = statistical_analysis(pulsar_df, ["P0", "DM"])

        assert result.success
        assert len(result.summaries) == 2
        params = [s.parameter for s in result.summaries]
        assert "P0" in params
        assert "DM" in params

    def test_summary_values(self, pulsar_df: pd.DataFrame) -> None:
        """Test that summary contains expected values."""
        result = statistical_analysis(pulsar_df, ["P0"])

        summary = result.summaries[0]
        assert summary.count == 100
        assert summary.mean > 0
        assert summary.std > 0
        assert summary.min < summary.max
        assert summary.q25 < summary.median < summary.q75


class TestCorrelationAnalysis:
    """Tests for correlation_analysis function."""

    @pytest.fixture
    def correlated_df(self) -> pd.DataFrame:
        """Create DataFrame with correlated variables."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.5  # Strong positive correlation
        z = -x + np.random.randn(100) * 0.5  # Strong negative correlation
        return pd.DataFrame({"X": x, "Y": y, "Z": z})

    def test_correlation_positive(self, correlated_df: pd.DataFrame) -> None:
        """Test detecting positive correlation."""
        result = correlation_analysis(correlated_df, "X", "Y")

        assert result.success
        assert len(result.correlations) == 1
        corr = result.correlations[0]
        assert corr.pearson_r > 0.9  # Strong positive
        assert corr.pearson_p < 0.01  # Significant

    def test_correlation_negative(self, correlated_df: pd.DataFrame) -> None:
        """Test detecting negative correlation."""
        result = correlation_analysis(correlated_df, "X", "Z")

        assert result.success
        corr = result.correlations[0]
        assert corr.pearson_r < -0.7  # Strong negative

    def test_correlation_log_transform(self) -> None:
        """Test log transform for power-law relationships."""
        np.random.seed(42)
        x = np.exp(np.random.randn(100))
        y = x ** 2 * np.exp(np.random.randn(100) * 0.2)
        df = pd.DataFrame({"X": x, "Y": y})

        result = correlation_analysis(df, "X", "Y", use_log=True)

        assert result.success
        corr = result.correlations[0]
        assert "(log)" in corr.param_x

    def test_correlation_interpretation(self, correlated_df: pd.DataFrame) -> None:
        """Test that interpretation is generated."""
        result = correlation_analysis(correlated_df, "X", "Y")

        corr = result.correlations[0]
        assert len(corr.interpretation) > 0
        assert "strong" in corr.interpretation.lower()

    def test_correlation_insufficient_data(self) -> None:
        """Test handling of insufficient data."""
        df = pd.DataFrame({"X": [1, 2], "Y": [3, np.nan]})  # Only 1 complete pair

        result = correlation_analysis(df, "X", "Y")

        assert not result.success
        assert "insufficient" in result.error.lower()


class TestToolDefinitions:
    """Tests for LLM tool definitions."""

    def test_get_tools_for_claude(self) -> None:
        """Test getting tool definitions."""
        tools = get_tools_for_claude()

        assert len(tools) >= 5
        assert all("name" in t for t in tools)
        assert all("description" in t for t in tools)
        assert all("input_schema" in t for t in tools)

    def test_get_tool_names(self) -> None:
        """Test getting tool names."""
        names = get_tool_names()

        assert "query_catalogue" in names
        assert "get_pulsar_info" in names
        assert "compute_derived_parameter" in names
        assert "statistical_analysis" in names

    def test_tool_schema_format(self) -> None:
        """Test that tool schemas are valid JSON Schema."""
        tools = get_tools_for_claude()

        for tool in tools:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
