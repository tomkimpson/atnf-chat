"""Tests for the Catalogue Interface module."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from atnf_chat.core.catalogue import (
    CatalogueInterface,
    CatalogueMetadata,
    QueryProvenance,
    get_catalogue,
)
from atnf_chat.core.dsl import (
    ComparisonOp,
    FilterClause,
    FilterGroup,
    LogicalOp,
    QueryDSL,
)


class TestCatalogueMetadata:
    """Tests for CatalogueMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Test creating catalogue metadata."""
        metadata = CatalogueMetadata(
            version="2.0.0",
            snapshot_date="2025-01-15T10:00:00",
            total_pulsars=3500,
            available_params=("JNAME", "P0", "DM"),
            param_coverage={"JNAME": 3500, "P0": 3400, "DM": 3300},
        )
        assert metadata.version == "2.0.0"
        assert metadata.total_pulsars == 3500
        assert len(metadata.available_params) == 3

    def test_metadata_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        metadata = CatalogueMetadata(
            version="2.0.0",
            snapshot_date="2025-01-15T10:00:00",
            total_pulsars=100,
            available_params=("JNAME", "P0"),
        )
        d = metadata.to_dict()
        assert d["version"] == "2.0.0"
        assert d["total_pulsars"] == 100
        assert "JNAME" in d["available_params"]


class TestQueryProvenance:
    """Tests for QueryProvenance dataclass."""

    def test_create_provenance(self) -> None:
        """Test creating query provenance."""
        prov = QueryProvenance(
            catalogue_version="2.0.0",
            snapshot_date="2025-01-15T10:00:00",
            query_dsl={"select_fields": ["JNAME"]},
            condition_string="P0 < 0.03",
            result_count=100,
            null_counts={"P0": 5, "DM": 10},
            completeness={"P0": 0.95, "DM": 0.90},
            timestamp="2025-01-15T10:00:01",
            query_hash="abc123",
        )
        assert prov.result_count == 100
        assert prov.null_counts["P0"] == 5

    def test_provenance_to_dict(self) -> None:
        """Test converting provenance to dictionary."""
        prov = QueryProvenance(
            catalogue_version="2.0.0",
            snapshot_date="2025-01-15T10:00:00",
            query_dsl={},
            condition_string=None,
            result_count=50,
            null_counts={},
            completeness={},
            timestamp="2025-01-15T10:00:01",
            query_hash="abc",
        )
        d = prov.to_dict()
        assert d["result_count"] == 50
        assert d["query_hash"] == "abc"

    def test_get_high_missingness_fields(self) -> None:
        """Test identifying fields with high missingness."""
        prov = QueryProvenance(
            catalogue_version="2.0.0",
            snapshot_date="2025-01-15T10:00:00",
            query_dsl={},
            condition_string=None,
            result_count=100,
            null_counts={"P0": 5, "ECC": 60, "PX": 80},
            completeness={"P0": 0.95, "ECC": 0.40, "PX": 0.20},
            timestamp="2025-01-15T10:00:01",
            query_hash="abc",
        )
        # Default threshold 0.5 means fields with >50% missing (completeness < 50%)
        high_miss = prov.get_high_missingness_fields()
        assert "P0" not in high_miss  # 95% complete, 5% missing
        assert "ECC" in high_miss  # 40% complete, 60% missing > 50%
        assert "PX" in high_miss  # 20% complete, 80% missing > 50%

        # With higher threshold 0.7 = fields with >70% missing
        high_miss = prov.get_high_missingness_fields(threshold=0.7)
        assert "P0" not in high_miss  # 5% missing < 70%
        assert "ECC" not in high_miss  # 60% missing < 70%
        assert "PX" in high_miss  # 80% missing > 70%


class TestCatalogueInterfaceUnit:
    """Unit tests for CatalogueInterface (with mocking)."""

    @pytest.fixture
    def mock_catalogue_df(self) -> pd.DataFrame:
        """Create a mock catalogue DataFrame."""
        return pd.DataFrame({
            "JNAME": ["J0030+0451", "J0437-4715", "J1012+5307", "J1713+0747", "J1909-3744"],
            "BNAME": ["B0030+0451", None, None, None, None],
            "P0": [0.00487, 0.00576, 0.00525, 0.00457, 0.00285],
            "P1": [1.0e-20, 5.7e-20, 1.7e-20, 9.3e-21, 1.4e-20],
            "DM": [4.33, 2.64, 9.02, 15.99, 10.39],
            "BSURF": [1.5e8, 2.9e8, 1.9e8, 1.3e8, 1.2e8],
            "EDOT": [3.5e33, 5.5e33, 4.3e33, 3.4e33, 4.3e33],
            "PB": [None, 5.74, 0.60, 67.83, 1.53],
            "ECC": [None, 1.9e-5, 1.2e-6, 7.5e-5, 1.4e-7],
            "ASSOC": [None, None, None, None, "GC:NGC6544"],
        })

    @pytest.fixture
    def mock_interface(self, mock_catalogue_df: pd.DataFrame) -> CatalogueInterface:
        """Create a CatalogueInterface with mocked psrqpy."""
        with patch("psrqpy.QueryATNF") as mock_query:
            mock_query_instance = MagicMock()
            mock_query_instance.pandas = mock_catalogue_df
            mock_query_instance.catalogue_version = "2.0.0"
            mock_query.return_value = mock_query_instance

            interface = CatalogueInterface()
            return interface

    def test_catalogue_loads_correctly(self, mock_interface: CatalogueInterface) -> None:
        """Test that catalogue loads and has correct metadata."""
        assert mock_interface.total_pulsars == 5
        assert mock_interface.version is not None
        assert "JNAME" in mock_interface.metadata.available_params

    def test_catalogue_dataframe_accessible(
        self, mock_interface: CatalogueInterface
    ) -> None:
        """Test that DataFrame is accessible."""
        df = mock_interface.dataframe
        assert len(df) == 5
        assert "P0" in df.columns

    def test_get_catalogue_info(self, mock_interface: CatalogueInterface) -> None:
        """Test get_catalogue_info method."""
        info = mock_interface.get_catalogue_info()
        assert "version" in info
        assert "total_pulsars" in info
        assert info["total_pulsars"] == 5

    def test_get_param_coverage(self, mock_interface: CatalogueInterface) -> None:
        """Test parameter coverage calculation."""
        coverage = mock_interface.get_param_coverage(["P0", "PB", "ECC"])
        assert coverage["P0"] == 1.0  # All have P0
        assert coverage["PB"] == 0.8  # 4/5 have PB
        assert coverage["ECC"] == 0.8  # 4/5 have ECC

    def test_query_simple(self, mock_interface: CatalogueInterface) -> None:
        """Test simple query execution."""
        query = QueryDSL(select_fields=["JNAME", "P0"])
        results, provenance = mock_interface.query(query)

        assert len(results) == 5
        assert list(results.columns) == ["JNAME", "P0"]
        assert provenance.result_count == 5
        assert provenance.catalogue_version == mock_interface.version

    def test_query_with_filter(self, mock_interface: CatalogueInterface) -> None:
        """Test query with filter."""
        query = QueryDSL(
            select_fields=["JNAME", "P0"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[FilterClause(field="P0", cmp=ComparisonOp.LT, value=0.005)],
            ),
        )
        results, provenance = mock_interface.query(query)

        # Only pulsars with P0 < 0.005
        assert len(results) < 5
        assert all(results["P0"] < 0.005)

    def test_query_with_ordering(self, mock_interface: CatalogueInterface) -> None:
        """Test query with ordering."""
        query = QueryDSL(
            select_fields=["JNAME", "P0"],
            order_by="P0",
            order_desc=False,
        )
        results, _ = mock_interface.query(query)

        # Check ascending order
        assert list(results["P0"]) == sorted(results["P0"])

    def test_query_with_limit(self, mock_interface: CatalogueInterface) -> None:
        """Test query with limit."""
        query = QueryDSL(select_fields=["JNAME"], limit=3)
        results, provenance = mock_interface.query(query)

        assert len(results) == 3
        assert provenance.result_count == 3

    def test_query_contains_filter(self, mock_interface: CatalogueInterface) -> None:
        """Test query with contains filter."""
        query = QueryDSL(
            select_fields=["JNAME", "ASSOC"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[
                    FilterClause(field="ASSOC", cmp=ComparisonOp.CONTAINS, value="GC")
                ],
            ),
        )
        results, _ = mock_interface.query(query)

        # Only J1909-3744 has GC association
        assert len(results) == 1
        assert "GC" in str(results.iloc[0]["ASSOC"])

    def test_query_not_null_filter(self, mock_interface: CatalogueInterface) -> None:
        """Test query with NOT NULL filter."""
        query = QueryDSL(
            select_fields=["JNAME", "PB"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[FilterClause(field="PB", cmp=ComparisonOp.NOT_NULL)],
            ),
        )
        results, _ = mock_interface.query(query)

        # 4 pulsars have PB (all except first one)
        assert len(results) == 4
        assert results["PB"].notna().all()

    def test_query_range_filter(self, mock_interface: CatalogueInterface) -> None:
        """Test query with range filter."""
        query = QueryDSL(
            select_fields=["JNAME", "P0"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[
                    FilterClause(
                        field="P0", cmp=ComparisonOp.IN_RANGE, value=[0.004, 0.005]
                    )
                ],
            ),
        )
        results, _ = mock_interface.query(query)

        assert all((results["P0"] >= 0.004) & (results["P0"] <= 0.005))

    def test_query_complex_filter(self, mock_interface: CatalogueInterface) -> None:
        """Test query with complex nested filters."""
        query = QueryDSL(
            select_fields=["JNAME", "P0", "PB"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[
                    FilterClause(field="P0", cmp=ComparisonOp.LT, value=0.006),
                    FilterGroup(
                        op=LogicalOp.OR,
                        clauses=[
                            FilterClause(field="PB", cmp=ComparisonOp.LT, value=2.0),
                            FilterClause(field="PB", cmp=ComparisonOp.IS_NULL),
                        ],
                    ),
                ],
            ),
        )
        results, _ = mock_interface.query(query)

        # P0 < 0.006 AND (PB < 2.0 OR PB is null)
        assert all(results["P0"] < 0.006)

    def test_query_provenance_null_counts(
        self, mock_interface: CatalogueInterface
    ) -> None:
        """Test that provenance includes null counts."""
        query = QueryDSL(select_fields=["JNAME", "PB", "ECC", "ASSOC"])
        _, provenance = mock_interface.query(query)

        assert "PB" in provenance.null_counts
        assert "ECC" in provenance.null_counts
        # ASSOC has 4 nulls out of 5
        assert provenance.null_counts["ASSOC"] == 4

    def test_query_caching(self, mock_interface: CatalogueInterface) -> None:
        """Test that identical queries are cached."""
        query = QueryDSL(select_fields=["JNAME", "P0"])

        # First query
        results1, prov1 = mock_interface.query(query)

        # Second identical query (should be cached)
        results2, prov2 = mock_interface.query(query)

        assert prov1.query_hash == prov2.query_hash
        assert results1.equals(results2)

    def test_clear_cache(self, mock_interface: CatalogueInterface) -> None:
        """Test clearing the query cache."""
        query = QueryDSL(select_fields=["JNAME"])
        mock_interface.query(query)

        assert len(mock_interface._query_cache) > 0
        mock_interface.clear_cache()
        assert len(mock_interface._query_cache) == 0

    def test_get_pulsar_by_jname(self, mock_interface: CatalogueInterface) -> None:
        """Test getting single pulsar by JNAME."""
        pulsar = mock_interface.get_pulsar("J0437-4715")
        assert pulsar is not None
        assert pulsar["JNAME"] == "J0437-4715"

    def test_get_pulsar_by_bname(self, mock_interface: CatalogueInterface) -> None:
        """Test getting single pulsar by BNAME."""
        pulsar = mock_interface.get_pulsar("B0030+0451")
        assert pulsar is not None
        assert pulsar["BNAME"] == "B0030+0451"

    def test_get_pulsar_not_found(self, mock_interface: CatalogueInterface) -> None:
        """Test getting non-existent pulsar returns None."""
        pulsar = mock_interface.get_pulsar("J9999+9999")
        assert pulsar is None

    def test_search_pulsars(self, mock_interface: CatalogueInterface) -> None:
        """Test searching for pulsars by pattern."""
        results = mock_interface.search_pulsars("J1")
        # Should find pulsars starting with J1
        assert len(results) > 0
        assert all(results["JNAME"].str.contains("J1"))


@pytest.mark.integration
class TestCatalogueInterfaceIntegration:
    """Integration tests for CatalogueInterface (requires network)."""

    @pytest.fixture(scope="class")
    def catalogue(self) -> CatalogueInterface:
        """Load the real catalogue (cached across tests in this class)."""
        return CatalogueInterface()

    def test_real_catalogue_loads(self, catalogue: CatalogueInterface) -> None:
        """Test that real catalogue loads successfully."""
        assert catalogue.total_pulsars > 3000  # Should have many pulsars
        assert catalogue.version != "unknown"

    def test_real_catalogue_has_expected_params(
        self, catalogue: CatalogueInterface
    ) -> None:
        """Test that expected parameters are present."""
        params = catalogue.metadata.available_params
        expected = ["JNAME", "P0", "P1", "F0", "DM", "RAJD", "DECJD"]
        for param in expected:
            assert param in params, f"Missing expected parameter: {param}"

    def test_real_msp_query(self, catalogue: CatalogueInterface) -> None:
        """Test querying for millisecond pulsars."""
        query = QueryDSL(
            select_fields=["JNAME", "P0", "DM"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[FilterClause(field="P0", cmp=ComparisonOp.LT, value=0.03)],
            ),
            order_by="P0",
            limit=100,
        )
        results, provenance = catalogue.query(query)

        assert len(results) > 0
        assert all(results["P0"] < 0.03)
        assert provenance.catalogue_version == catalogue.version

    def test_real_binary_pulsar_query(self, catalogue: CatalogueInterface) -> None:
        """Test querying for binary pulsars."""
        query = QueryDSL(
            select_fields=["JNAME", "P0", "PB", "ECC"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[FilterClause(field="PB", cmp=ComparisonOp.NOT_NULL)],
            ),
            limit=50,
        )
        results, provenance = catalogue.query(query)

        assert len(results) > 0
        assert results["PB"].notna().all()

    def test_real_gc_pulsar_query(self, catalogue: CatalogueInterface) -> None:
        """Test querying for globular cluster pulsars."""
        query = QueryDSL(
            select_fields=["JNAME", "P0", "ASSOC"],
            filters=FilterGroup(
                op=LogicalOp.AND,
                clauses=[
                    FilterClause(field="ASSOC", cmp=ComparisonOp.CONTAINS, value="GC")
                ],
            ),
        )
        results, _ = catalogue.query(query)

        # Should find some GC pulsars
        assert len(results) > 0

    def test_real_known_pulsar_lookup(self, catalogue: CatalogueInterface) -> None:
        """Test looking up a known pulsar."""
        # Vela pulsar
        pulsar = catalogue.get_pulsar("J0835-4510")
        if pulsar is not None:
            assert abs(pulsar["P0"] - 0.089) < 0.001  # ~89ms period


class TestGetCatalogue:
    """Tests for the get_catalogue singleton function."""

    def test_get_catalogue_returns_instance(self) -> None:
        """Test that get_catalogue returns an instance."""
        with patch("psrqpy.QueryATNF") as mock_query:
            mock_query_instance = MagicMock()
            mock_query_instance.pandas = pd.DataFrame({"JNAME": ["J0001+0000"]})
            mock_query_instance.catalogue_version = "2.0.0"
            mock_query.return_value = mock_query_instance

            # Reset the singleton
            import atnf_chat.core.catalogue as cat_module
            cat_module._catalogue_instance = None

            result = get_catalogue()
            assert result is not None
            assert isinstance(result, CatalogueInterface)

    def test_get_catalogue_singleton(self) -> None:
        """Test that get_catalogue returns same instance."""
        with patch("psrqpy.QueryATNF") as mock_query:
            mock_query_instance = MagicMock()
            mock_query_instance.pandas = pd.DataFrame({"JNAME": ["J0001+0000"]})
            mock_query_instance.catalogue_version = "2.0.0"
            mock_query.return_value = mock_query_instance

            # Reset the singleton
            import atnf_chat.core.catalogue as cat_module
            cat_module._catalogue_instance = None

            result1 = get_catalogue()
            result2 = get_catalogue()

            # Should only create one instance
            assert mock_query.call_count == 1
            assert result1 is result2
