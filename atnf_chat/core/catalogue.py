"""Catalogue Interface for ATNF Pulsar Catalogue.

This module provides the data layer for interacting with the ATNF Pulsar Catalogue
via psrqpy. It handles:
- Catalogue loading and caching
- Version detection and tracking
- Query execution with provenance
- Result validation and null tracking

Example:
    >>> from atnf_chat.core.catalogue import CatalogueInterface
    >>> catalogue = CatalogueInterface()
    >>> print(f"Loaded {catalogue.total_pulsars} pulsars from v{catalogue.version}")

    >>> from atnf_chat.core.dsl import QueryDSL, FilterGroup, FilterClause
    >>> query = QueryDSL(
    ...     select_fields=["JNAME", "P0", "DM"],
    ...     filters=FilterGroup(op="and", clauses=[
    ...         FilterClause(field="P0", cmp="lt", value=0.03)
    ...     ])
    ... )
    >>> results, provenance = catalogue.query(query)
    >>> print(f"Found {len(results)} MSPs")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from atnf_chat.core.dsl import QueryDSL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogueMetadata:
    """Metadata about the loaded catalogue.

    Attributes:
        version: Catalogue version string (e.g., "2.0.0")
        snapshot_date: When the catalogue was loaded
        total_pulsars: Number of pulsars in the catalogue
        available_params: List of available parameter codes
        param_coverage: Dict mapping param -> count of non-null values
    """

    version: str
    snapshot_date: str
    total_pulsars: int
    available_params: tuple[str, ...]
    param_coverage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "snapshot_date": self.snapshot_date,
            "total_pulsars": self.total_pulsars,
            "available_params": list(self.available_params),
            "param_coverage": self.param_coverage,
        }


@dataclass
class QueryProvenance:
    """Provenance information for a query result.

    This tracks all metadata needed for reproducibility and scientific safety.

    Attributes:
        catalogue_version: Version of the catalogue used
        snapshot_date: When catalogue was loaded
        query_dsl: The query that was executed (as dict)
        condition_string: The psrqpy condition string
        result_count: Number of rows returned
        null_counts: Dict mapping field -> count of null values
        completeness: Dict mapping field -> fraction of non-null values
        timestamp: When the query was executed
        query_hash: Hash of query for caching/deduplication
    """

    catalogue_version: str
    snapshot_date: str
    query_dsl: dict[str, Any]
    condition_string: str | None
    result_count: int
    null_counts: dict[str, int]
    completeness: dict[str, float]
    timestamp: str
    query_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "catalogue_version": self.catalogue_version,
            "snapshot_date": self.snapshot_date,
            "query_dsl": self.query_dsl,
            "condition_string": self.condition_string,
            "result_count": self.result_count,
            "null_counts": self.null_counts,
            "completeness": self.completeness,
            "timestamp": self.timestamp,
            "query_hash": self.query_hash,
        }

    def get_high_missingness_fields(self, threshold: float = 0.5) -> list[str]:
        """Get fields where missingness exceeds threshold.

        Args:
            threshold: Fraction of missing values to flag (default: 0.5 = 50%)

        Returns:
            List of field names with high missingness
        """
        return [
            field
            for field, completeness in self.completeness.items()
            if completeness < (1.0 - threshold)
        ]


class CatalogueInterface:
    """Interface to the ATNF Pulsar Catalogue.

    This class provides a high-level interface for querying the catalogue
    with full provenance tracking and validation.

    Attributes:
        metadata: CatalogueMetadata with version and stats
        version: Catalogue version string
        total_pulsars: Number of pulsars in catalogue

    Example:
        >>> catalogue = CatalogueInterface()
        >>> print(catalogue.version)
        '2.0.0'
        >>> print(catalogue.total_pulsars)
        3389
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        force_refresh: bool = False,
    ) -> None:
        """Initialize the catalogue interface.

        Args:
            cache_dir: Directory for caching catalogue data (optional)
            force_refresh: Force download of fresh catalogue data
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._force_refresh = force_refresh
        self._df: pd.DataFrame | None = None
        self._metadata: CatalogueMetadata | None = None
        self._query_cache: dict[str, tuple[pd.DataFrame, QueryProvenance]] = {}

        # Load catalogue on init
        self._load_catalogue()

    def _load_catalogue(self) -> None:
        """Load the catalogue from psrqpy."""
        from psrqpy import QueryATNF

        logger.info("Loading ATNF Pulsar Catalogue...")
        start_time = datetime.now()

        try:
            # Load full catalogue
            # psrqpy caches locally by default
            # Note: loadfromdb expects a path string or None, not a boolean
            # Passing None uses the default cached database
            query = QueryATNF(loadfromdb=None if not self._force_refresh else True)
            self._df = query.pandas

            # Extract metadata
            version = self._detect_version(query)
            snapshot_date = datetime.now().isoformat()
            total_pulsars = len(self._df)
            available_params = tuple(self._df.columns.tolist())

            # Calculate parameter coverage
            param_coverage = {
                col: int(self._df[col].notna().sum()) for col in self._df.columns
            }

            self._metadata = CatalogueMetadata(
                version=version,
                snapshot_date=snapshot_date,
                total_pulsars=total_pulsars,
                available_params=available_params,
                param_coverage=param_coverage,
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Loaded {total_pulsars} pulsars (v{version}) in {elapsed:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load catalogue: {e}")
            raise RuntimeError(f"Failed to load ATNF catalogue: {e}") from e

    def _detect_version(self, query: Any) -> str:
        """Detect catalogue version from psrqpy query object.

        Args:
            query: psrqpy QueryATNF object

        Returns:
            Version string or 'unknown'
        """
        # psrqpy stores version info in different attributes depending on version
        version_attrs = ["catalogue_version", "version", "catversion"]

        for attr in version_attrs:
            if hasattr(query, attr):
                version = getattr(query, attr)
                if version:
                    return str(version)

        # Try to get from catalogue data if available
        if hasattr(query, "catalogue") and query.catalogue is not None:
            cat = query.catalogue
            if hasattr(cat, "version"):
                return str(cat.version)

        logger.warning("Could not detect catalogue version")
        return "unknown"

    @property
    def metadata(self) -> CatalogueMetadata:
        """Get catalogue metadata."""
        if self._metadata is None:
            raise RuntimeError("Catalogue not loaded")
        return self._metadata

    @property
    def version(self) -> str:
        """Get catalogue version string."""
        return self.metadata.version

    @property
    def total_pulsars(self) -> int:
        """Get total number of pulsars in catalogue."""
        return self.metadata.total_pulsars

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the full catalogue DataFrame."""
        if self._df is None:
            raise RuntimeError("Catalogue not loaded")
        return self._df

    def get_catalogue_info(self) -> dict[str, Any]:
        """Get catalogue information for display or LLM context.

        Returns:
            Dictionary with catalogue metadata
        """
        return self.metadata.to_dict()

    def get_param_coverage(self, params: list[str] | None = None) -> dict[str, float]:
        """Get coverage (non-null fraction) for parameters.

        Args:
            params: List of parameter codes (None = all)

        Returns:
            Dict mapping param -> coverage fraction
        """
        coverage = self.metadata.param_coverage
        total = self.metadata.total_pulsars

        if params is None:
            params = list(coverage.keys())

        return {p: coverage.get(p, 0) / total for p in params if p in coverage}

    def query(
        self,
        query_dsl: QueryDSL,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, QueryProvenance]:
        """Execute a query and return results with provenance.

        Args:
            query_dsl: Validated QueryDSL object
            use_cache: Whether to use cached results for identical queries

        Returns:
            Tuple of (results DataFrame, provenance)
        """
        # Generate query hash for caching
        query_hash = self._hash_query(query_dsl)

        # Check cache
        if use_cache and query_hash in self._query_cache:
            logger.debug(f"Cache hit for query {query_hash[:8]}")
            cached_df, cached_prov = self._query_cache[query_hash]
            # Update timestamp for cached result
            return cached_df.copy(), QueryProvenance(
                catalogue_version=cached_prov.catalogue_version,
                snapshot_date=cached_prov.snapshot_date,
                query_dsl=cached_prov.query_dsl,
                condition_string=cached_prov.condition_string,
                result_count=cached_prov.result_count,
                null_counts=cached_prov.null_counts,
                completeness=cached_prov.completeness,
                timestamp=datetime.now().isoformat(),
                query_hash=query_hash,
            )

        # Execute query
        results = self._execute_query(query_dsl)

        # Build provenance
        provenance = self._build_provenance(query_dsl, results, query_hash)

        # Cache results
        if use_cache:
            self._query_cache[query_hash] = (results.copy(), provenance)

        return results, provenance

    def _execute_query(self, query_dsl: QueryDSL) -> pd.DataFrame:
        """Execute query against the loaded catalogue.

        Args:
            query_dsl: Validated QueryDSL object

        Returns:
            DataFrame with query results
        """
        df = self.dataframe.copy()

        # Apply filters
        condition = query_dsl.to_psrqpy_condition()
        if condition:
            df = self._apply_condition(df, query_dsl)

        # Select fields
        params = query_dsl.to_psrqpy_params()
        if params:
            # Ensure all requested params exist
            available = [p for p in params if p in df.columns]
            missing = [p for p in params if p not in df.columns]
            if missing:
                logger.warning(f"Requested params not in catalogue: {missing}")
            df = df[available]

        # Apply ordering
        if query_dsl.order_by and query_dsl.order_by in df.columns:
            df = df.sort_values(
                query_dsl.order_by, ascending=not query_dsl.order_desc
            )

        # Apply limit
        if query_dsl.limit:
            df = df.head(query_dsl.limit)

        return df.reset_index(drop=True)

    def _apply_condition(self, df: pd.DataFrame, query_dsl: QueryDSL) -> pd.DataFrame:
        """Apply filter conditions to DataFrame.

        This reimplements the filtering in pandas rather than using psrqpy's
        condition string, for better control and error handling.

        Args:
            df: Input DataFrame
            query_dsl: Query with filters

        Returns:
            Filtered DataFrame
        """
        if query_dsl.filters is None:
            return df

        mask = self._evaluate_filter_group(df, query_dsl.filters)
        return df[mask]

    def _evaluate_filter_group(
        self, df: pd.DataFrame, group: Any
    ) -> pd.Series:
        """Evaluate a filter group to a boolean mask.

        Args:
            df: DataFrame to filter
            group: FilterGroup object

        Returns:
            Boolean Series mask
        """
        from atnf_chat.core.dsl import ComparisonOp, FilterClause, FilterGroup, LogicalOp

        if not isinstance(group, FilterGroup):
            raise TypeError(f"Expected FilterGroup, got {type(group)}")

        masks = []
        for clause in group.clauses:
            if isinstance(clause, FilterClause):
                mask = self._evaluate_clause(df, clause)
            elif isinstance(clause, FilterGroup):
                mask = self._evaluate_filter_group(df, clause)
            else:
                raise TypeError(f"Unexpected clause type: {type(clause)}")
            masks.append(mask)

        # Combine masks with logical operator
        if group.op == LogicalOp.AND:
            result = masks[0]
            for m in masks[1:]:
                result = result & m
            return result
        else:  # OR
            result = masks[0]
            for m in masks[1:]:
                result = result | m
            return result

    def _evaluate_clause(self, df: pd.DataFrame, clause: Any) -> pd.Series:
        """Evaluate a single filter clause to a boolean mask.

        Args:
            df: DataFrame to filter
            clause: FilterClause object

        Returns:
            Boolean Series mask
        """
        from atnf_chat.core.dsl import ComparisonOp, FilterClause

        if not isinstance(clause, FilterClause):
            raise TypeError(f"Expected FilterClause, got {type(clause)}")

        field = clause.field
        op = clause.cmp
        value = clause.value

        # Handle missing columns
        if field not in df.columns:
            logger.warning(f"Field {field} not in DataFrame, returning all False")
            return pd.Series([False] * len(df), index=df.index)

        col = df[field]

        # Null checks
        if op == ComparisonOp.IS_NULL:
            return col.isna()
        if op == ComparisonOp.NOT_NULL:
            return col.notna()

        # Comparison operators
        if op == ComparisonOp.EQ:
            return col == value
        if op == ComparisonOp.NE:
            return col != value
        if op == ComparisonOp.LT:
            return col < value
        if op == ComparisonOp.LE:
            return col <= value
        if op == ComparisonOp.GT:
            return col > value
        if op == ComparisonOp.GE:
            return col >= value

        # String operators
        if op == ComparisonOp.CONTAINS:
            # Case-insensitive contains
            return col.astype(str).str.contains(str(value), case=False, na=False)
        if op == ComparisonOp.STARTSWITH:
            return col.astype(str).str.startswith(str(value), na=False)

        # Range operator
        if op == ComparisonOp.IN_RANGE:
            min_val, max_val = value
            return (col >= min_val) & (col <= max_val)

        raise ValueError(f"Unknown operator: {op}")

    def _build_provenance(
        self,
        query_dsl: QueryDSL,
        results: pd.DataFrame,
        query_hash: str,
    ) -> QueryProvenance:
        """Build provenance information for query results.

        Args:
            query_dsl: The executed query
            results: Query results
            query_hash: Hash of the query

        Returns:
            QueryProvenance object
        """
        # Calculate null counts and completeness for result columns
        null_counts = {}
        completeness = {}
        result_count = len(results)

        for col in results.columns:
            null_count = int(results[col].isna().sum())
            null_counts[col] = null_count
            completeness[col] = (
                1.0 - (null_count / result_count) if result_count > 0 else 0.0
            )

        return QueryProvenance(
            catalogue_version=self.version,
            snapshot_date=self.metadata.snapshot_date,
            query_dsl=query_dsl.model_dump(mode="json"),
            condition_string=query_dsl.to_psrqpy_condition(),
            result_count=result_count,
            null_counts=null_counts,
            completeness=completeness,
            timestamp=datetime.now().isoformat(),
            query_hash=query_hash,
        )

    def _hash_query(self, query_dsl: QueryDSL) -> str:
        """Generate a hash for a query for caching.

        Args:
            query_dsl: Query to hash

        Returns:
            Hash string
        """
        import json

        query_json = json.dumps(query_dsl.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(query_json.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")

    def get_pulsar(self, name: str) -> pd.Series | None:
        """Get a single pulsar by name.

        Args:
            name: Pulsar name (JNAME, BNAME, or common name like 'Vela', 'Crab')

        Returns:
            Series with pulsar data, or None if not found
        """
        df = self.dataframe

        # Try JNAME first (exact match)
        if "JNAME" in df.columns:
            matches = df[df["JNAME"] == name]
            if len(matches) == 1:
                return matches.iloc[0]

        # Try BNAME (exact match)
        if "BNAME" in df.columns:
            matches = df[df["BNAME"] == name]
            if len(matches) == 1:
                return matches.iloc[0]

        # Try partial match on JNAME
        if "JNAME" in df.columns:
            matches = df[df["JNAME"].str.contains(name, case=False, na=False)]
            if len(matches) == 1:
                return matches.iloc[0]

        # Try searching ASSOC_ORIG for common names (e.g., "Vela", "Crab")
        # These appear as "SNR:Vela" or "SNR:Crab" in the associations
        if "ASSOC_ORIG" in df.columns:
            # Search for the name as a word boundary in associations
            # This matches "SNR:Vela" but not "VelaX" in other contexts
            pattern = rf"(?:^|[,:]){name}(?:[,\[]|$)"
            matches = df[df["ASSOC_ORIG"].str.contains(pattern, case=False, na=False, regex=True)]
            if len(matches) == 1:
                return matches.iloc[0]

        return None

    def search_pulsars(
        self,
        name_pattern: str,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Search for pulsars by name pattern.

        Searches JNAME, BNAME, and ASSOC_ORIG columns for matches.
        This allows finding pulsars by common names like "Vela" or "Crab".

        Args:
            name_pattern: Pattern to search for
            limit: Maximum results to return

        Returns:
            DataFrame with matching pulsars
        """
        df = self.dataframe
        results = []

        # Search JNAME
        if "JNAME" in df.columns:
            matches = df[
                df["JNAME"].str.contains(name_pattern, case=False, na=False)
            ]
            results.append(matches)

        # Search BNAME
        if "BNAME" in df.columns:
            matches = df[
                df["BNAME"].str.contains(name_pattern, case=False, na=False)
            ]
            results.append(matches)

        # Search ASSOC_ORIG for common names (e.g., "Vela", "Crab")
        if "ASSOC_ORIG" in df.columns:
            matches = df[
                df["ASSOC_ORIG"].str.contains(name_pattern, case=False, na=False)
            ]
            results.append(matches)

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results).drop_duplicates()
        return combined.head(limit)


# Singleton instance for convenience
_catalogue_instance: CatalogueInterface | None = None


def get_catalogue(force_new: bool = False, **kwargs: Any) -> CatalogueInterface:
    """Get the shared catalogue instance.

    This provides a singleton pattern for the catalogue to avoid
    reloading it multiple times.

    Args:
        force_new: Force creation of a new instance
        **kwargs: Arguments passed to CatalogueInterface

    Returns:
        CatalogueInterface instance
    """
    global _catalogue_instance

    if _catalogue_instance is None or force_new:
        _catalogue_instance = CatalogueInterface(**kwargs)

    return _catalogue_instance
