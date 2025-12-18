"""Tests for the FastAPI application."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from atnf_chat.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_catalogue() -> MagicMock:
    """Create a mock catalogue interface."""
    mock = MagicMock()
    mock.version = "2.0.0"

    # Mock DataFrame
    df = pd.DataFrame({
        "JNAME": ["J0030+0451", "J0437-4715", "J1012+5307"],
        "P0": [0.00487, 0.00576, 0.00525],
        "P1": [1.0e-20, 5.7e-20, 1.7e-20],
        "DM": [4.33, 2.64, 9.02],
        "DIST": [0.3, 0.16, 0.52],
        "RAJD": [7.625, 69.316, 153.140],
        "DECJD": [4.852, -47.253, 53.117],
        "GL": [111.5, 253.4, 160.4],
        "GB": [-57.6, -41.9, 58.9],
        "TYPE": ["MSP", "MSP", "MSP"],
    })
    mock._df = df

    # Mock provenance
    provenance = MagicMock()
    provenance.catalogue_version = "2.0.0"
    provenance.result_count = 3
    provenance.null_counts = {col: 0 for col in df.columns}
    provenance.completeness = {col: 1.0 for col in df.columns}
    provenance.to_dict.return_value = {
        "catalogue_version": "2.0.0",
        "result_count": 3,
    }

    mock.query.return_value = (df, provenance)
    mock.get_pulsar.return_value = df.iloc[0]

    return mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check returns status."""
        with patch("atnf_chat.api.app.get_catalogue") as mock_get:
            mock_cat = MagicMock()
            mock_cat.version = "2.0.0"
            mock_cat._df = pd.DataFrame({"JNAME": ["J0001"] * 100})
            mock_get.return_value = mock_cat

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data


class TestQueryEndpoint:
    """Tests for query endpoint."""

    def test_query_valid_dsl(
        self, client: TestClient, mock_catalogue: MagicMock
    ) -> None:
        """Test query with valid DSL."""
        with patch("atnf_chat.api.app._catalogue", mock_catalogue):
            with patch("atnf_chat.api.app.query_catalogue") as mock_query:
                # Create a mock result
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.data = mock_catalogue._df
                mock_result.error = None
                mock_result.provenance = MagicMock()
                mock_result.provenance.to_dict.return_value = {}
                mock_query.return_value = mock_result

                response = client.post(
                    "/query",
                    json={
                        "query_dsl": {
                            "select_fields": ["JNAME", "P0"],
                            "limit": 10,
                        }
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"]

    def test_query_invalid_dsl(self, client: TestClient) -> None:
        """Test query with invalid DSL."""
        with patch("atnf_chat.api.app.query_catalogue") as mock_query:
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Invalid field: NONEXISTENT"
            mock_result.data = None
            mock_query.return_value = mock_result

            response = client.post(
                "/query",
                json={
                    "query_dsl": {
                        "select_fields": ["NONEXISTENT"],
                    }
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert not data["success"]
            assert "NONEXISTENT" in data["error"]


class TestPulsarEndpoint:
    """Tests for pulsar info endpoint."""

    def test_get_pulsar_info(
        self, client: TestClient, mock_catalogue: MagicMock
    ) -> None:
        """Test getting pulsar information."""
        with patch("atnf_chat.api.app._catalogue", mock_catalogue):
            with patch("atnf_chat.api.app.get_pulsar_info") as mock_get:
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {
                    "success": True,
                    "data": {"JNAME": "J0437-4715", "P0": 0.00576},
                }
                mock_get.return_value = mock_result

                response = client.post(
                    "/pulsar",
                    json={"pulsar_name": "J0437-4715"},
                )

                assert response.status_code == 200


class TestAnalysisEndpoints:
    """Tests for analysis endpoints."""

    def test_statistical_analysis(
        self, client: TestClient, mock_catalogue: MagicMock
    ) -> None:
        """Test statistical analysis endpoint."""
        with patch("atnf_chat.api.app._catalogue", mock_catalogue):
            with patch("atnf_chat.api.app.query_catalogue") as mock_query:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.data = mock_catalogue._df
                mock_result.error = None
                mock_query.return_value = mock_result

                with patch("atnf_chat.api.app.statistical_analysis") as mock_stats:
                    mock_stats_result = MagicMock()
                    mock_stats_result.to_dict.return_value = {
                        "success": True,
                        "summaries": [],
                    }
                    mock_stats.return_value = mock_stats_result

                    response = client.post(
                        "/analysis/statistics",
                        json={
                            "query_dsl": {"select_fields": ["P0", "DM"]},
                            "parameters": ["P0"],
                        },
                    )

                    assert response.status_code == 200

    def test_correlation_analysis(
        self, client: TestClient, mock_catalogue: MagicMock
    ) -> None:
        """Test correlation analysis endpoint."""
        with patch("atnf_chat.api.app._catalogue", mock_catalogue):
            with patch("atnf_chat.api.app.query_catalogue") as mock_query:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.data = mock_catalogue._df
                mock_result.error = None
                mock_query.return_value = mock_result

                with patch("atnf_chat.api.app.correlation_analysis") as mock_corr:
                    mock_corr_result = MagicMock()
                    mock_corr_result.to_dict.return_value = {
                        "success": True,
                        "correlations": [],
                    }
                    mock_corr.return_value = mock_corr_result

                    response = client.post(
                        "/analysis/correlation",
                        json={
                            "query_dsl": {"select_fields": ["P0", "DM"]},
                            "param_x": "P0",
                            "param_y": "DM",
                        },
                    )

                    assert response.status_code == 200


class TestPlotEndpoint:
    """Tests for plot generation endpoint."""

    def test_generate_histogram(
        self, client: TestClient, mock_catalogue: MagicMock
    ) -> None:
        """Test generating a histogram plot."""
        with patch("atnf_chat.api.app._catalogue", mock_catalogue):
            with patch("atnf_chat.api.app.query_catalogue") as mock_query:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.data = mock_catalogue._df
                mock_result.error = None
                mock_query.return_value = mock_result

                with patch("atnf_chat.api.app.create_histogram") as mock_hist:
                    mock_fig = MagicMock()
                    mock_fig.to_json.return_value = "{}"
                    mock_fig.to_html.return_value = "<html></html>"

                    mock_plot_result = MagicMock()
                    mock_plot_result.success = True
                    mock_plot_result.figure = mock_fig
                    mock_plot_result.error = None
                    mock_hist.return_value = mock_plot_result

                    response = client.post(
                        "/plot",
                        json={
                            "plot_type": "histogram",
                            "query_dsl": {"select_fields": ["P0"]},
                            "options": {"parameter": "P0"},
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"]

    def test_generate_unknown_plot_type(
        self, client: TestClient, mock_catalogue: MagicMock
    ) -> None:
        """Test generating an unknown plot type."""
        with patch("atnf_chat.api.app._catalogue", mock_catalogue):
            with patch("atnf_chat.api.app.query_catalogue") as mock_query:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.data = mock_catalogue._df
                mock_result.error = None
                mock_query.return_value = mock_result

                response = client.post(
                    "/plot",
                    json={
                        "plot_type": "unknown_type",
                        "query_dsl": {"select_fields": ["P0"]},
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert not data["success"]
                assert "unknown" in data["error"].lower()


class TestCodeExportEndpoint:
    """Tests for code export endpoint."""

    def test_export_code(self, client: TestClient) -> None:
        """Test exporting Python code."""
        with patch("atnf_chat.api.app.generate_query_plan") as mock_plan:
            mock_plan.return_value = {
                "success": True,
                "plan": "Query millisecond pulsars",
                "python_code": "from psrqpy import QueryATNF\nq = QueryATNF()",
            }

            response = client.post(
                "/export/code",
                json={
                    "query_dsl": {
                        "select_fields": ["JNAME", "P0"],
                        "filters": {
                            "op": "and",
                            "clauses": [{"field": "P0", "cmp": "lt", "value": 0.03}],
                        },
                    }
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "python_code" in data
            assert "QueryATNF" in data["python_code"]

    def test_export_code_with_analysis(self, client: TestClient) -> None:
        """Test exporting code with analysis included."""
        with patch("atnf_chat.api.app.generate_query_plan") as mock_plan:
            mock_plan.return_value = {
                "success": True,
                "plan": "Query pulsars",
                "python_code": "from psrqpy import QueryATNF\nq = QueryATNF()",
            }

            response = client.post(
                "/export/code",
                json={
                    "query_dsl": {"select_fields": ["JNAME", "P0"]},
                    "include_analysis": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "Statistical analysis" in data["python_code"]


class TestCatalogueInfoEndpoint:
    """Tests for catalogue info endpoint."""

    def test_get_catalogue_info(
        self, client: TestClient, mock_catalogue: MagicMock
    ) -> None:
        """Test getting catalogue information."""
        with patch("atnf_chat.api.app._catalogue", mock_catalogue):
            with patch("atnf_chat.api.app.get_catalogue") as mock_get:
                mock_get.return_value = mock_catalogue

                response = client.get("/catalogue/info")

                assert response.status_code == 200
                data = response.json()
                assert "catalogue_version" in data
                assert "available_parameters" in data


class TestParameterInfoEndpoint:
    """Tests for parameter info endpoint."""

    def test_get_parameter_info_valid(self, client: TestClient) -> None:
        """Test getting info for a valid parameter."""
        response = client.get("/parameters/P0")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "P0"
        assert "description" in data
        assert "unit" in data

    def test_get_parameter_info_invalid(self, client: TestClient) -> None:
        """Test getting info for an invalid parameter."""
        response = client.get("/parameters/NONEXISTENT")

        assert response.status_code == 404
