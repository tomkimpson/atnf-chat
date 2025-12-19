"""Query tools for LLM function calling.

This module provides the main query interface for the LLM to interact
with the ATNF Pulsar Catalogue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import ValidationError

from atnf_chat.core.catalogue import CatalogueInterface, QueryProvenance, get_catalogue
from atnf_chat.core.dsl import QueryDSL
from atnf_chat.core.validation import ResultValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Complete result from a catalogue query.

    Attributes:
        success: Whether the query executed successfully
        data: Query results as DataFrame (None if failed)
        provenance: Query provenance information
        validation: Validation result with warnings
        error: Error message if query failed
        suggestions: Suggestions for fixing failed queries
    """

    success: bool
    data: pd.DataFrame | None = None
    provenance: QueryProvenance | None = None
    validation: ValidationResult | None = None
    error: str | None = None
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "success": self.success,
            "error": self.error,
            "suggestions": self.suggestions,
        }

        if self.data is not None:
            result["result_count"] = len(self.data)
            result["columns"] = list(self.data.columns)
            # Include first few rows as preview
            result["preview"] = self.data.head(10).to_dict(orient="records")

        if self.provenance is not None:
            result["provenance"] = self.provenance.to_dict()

        if self.validation is not None:
            result["validation"] = self.validation.to_dict()

        return result

    def format_for_display(self, max_rows: int = 20) -> str:
        """Format result for display in chat response.

        Args:
            max_rows: Maximum rows to display

        Returns:
            Formatted string representation
        """
        if not self.success:
            lines = [f"**Query Failed:** {self.error}"]
            if self.suggestions:
                lines.append("\n**Suggestions:**")
                for sug in self.suggestions:
                    lines.append(f"- {sug}")
            return "\n".join(lines)

        if self.data is None or len(self.data) == 0:
            return "No pulsars match your query criteria."

        lines = []

        # Header with metadata
        if self.provenance:
            lines.append(
                f"Found **{self.provenance.result_count}** pulsars "
                f"(catalogue v{self.provenance.catalogue_version})"
            )

        # Validation warnings
        if self.validation and self.validation.warnings:
            lines.append("")
            lines.append(self.validation.format_for_llm())

        # Data table
        lines.append("")
        display_df = self.data.head(max_rows)
        lines.append(display_df.to_markdown(index=False))

        if len(self.data) > max_rows:
            lines.append(f"\n*Showing first {max_rows} of {len(self.data)} results*")

        return "\n".join(lines)


def query_catalogue(
    query_dsl: dict[str, Any],
    catalogue: CatalogueInterface | None = None,
    validate: bool = True,
) -> QueryResult:
    """Execute a validated query against the ATNF catalogue.

    This is the main query function exposed to the LLM. It handles:
    - DSL validation with helpful error messages
    - Query execution with provenance tracking
    - Result validation with scientific safety checks
    - Error recovery suggestions

    Args:
        query_dsl: Query in DSL format (dict from LLM)
        catalogue: CatalogueInterface instance (uses singleton if None)
        validate: Whether to run result validation

    Returns:
        QueryResult with data, provenance, and validation

    Example:
        >>> result = query_catalogue({
        ...     "select_fields": ["JNAME", "P0", "DM"],
        ...     "filters": {
        ...         "op": "and",
        ...         "clauses": [{"field": "P0", "cmp": "lt", "value": 0.03}]
        ...     },
        ...     "limit": 100
        ... })
        >>> print(result.format_for_display())
    """
    # Validate DSL
    try:
        validated_query = QueryDSL(**query_dsl)
    except ValidationError as e:
        error_messages = []
        suggestions = []

        for error in e.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            msg = error["msg"]
            error_messages.append(f"{loc}: {msg}")

            # Generate helpful suggestions
            if "Unknown field" in msg:
                suggestions.append(
                    "Check parameter names. Common fields: "
                    "JNAME, P0, P1, F0, DM, BSURF, EDOT, AGE, PB, ECC, ASSOC"
                )
            elif "requires a value" in msg:
                suggestions.append(
                    "Comparison operators (lt, gt, eq, etc.) require a value"
                )
            elif "requires a [min, max] range" in msg:
                suggestions.append(
                    "The 'in_range' operator requires a list: [min_value, max_value]"
                )

        return QueryResult(
            success=False,
            error=f"Query validation failed: {'; '.join(error_messages)}",
            suggestions=suggestions or ["Check the query DSL format"],
        )

    # Get catalogue
    if catalogue is None:
        try:
            catalogue = get_catalogue()
        except Exception as e:
            return QueryResult(
                success=False,
                error=f"Failed to load catalogue: {e}",
                suggestions=["Check network connection or catalogue cache"],
            )

    # Execute query
    try:
        results, provenance = catalogue.query(validated_query)
    except Exception as e:
        logger.exception("Query execution failed")
        return QueryResult(
            success=False,
            error=f"Query execution failed: {e}",
            suggestions=["Try simplifying the query or check filter values"],
        )

    # Validate results
    validation = None
    if validate:
        validator = ResultValidator()
        validation = validator.validate(results, provenance, validated_query)

    return QueryResult(
        success=True,
        data=results,
        provenance=provenance,
        validation=validation,
    )


def get_pulsar_info(
    name: str,
    catalogue: CatalogueInterface | None = None,
    fields: list[str] | None = None,
) -> QueryResult:
    """Get detailed information about a specific pulsar.

    Args:
        name: Pulsar name (JNAME, BNAME, or partial match)
        catalogue: CatalogueInterface instance (uses singleton if None)
        fields: Optional list of specific fields to return

    Returns:
        QueryResult with pulsar data
    """
    if catalogue is None:
        try:
            catalogue = get_catalogue()
        except Exception as e:
            return QueryResult(
                success=False,
                error=f"Failed to load catalogue: {e}",
            )

    pulsar = catalogue.get_pulsar(name)

    if pulsar is None:
        # Try searching
        matches = catalogue.search_pulsars(name, limit=5)
        suggestions = []
        if len(matches) > 0:
            suggestions.append(
                f"Did you mean one of these? {', '.join(matches['JNAME'].tolist())}"
            )
        else:
            suggestions.append("Check the pulsar name spelling")

        return QueryResult(
            success=False,
            error=f"Pulsar '{name}' not found in catalogue",
            suggestions=suggestions,
        )

    # Convert Series to DataFrame for consistent interface
    df = pd.DataFrame([pulsar])

    # Filter to specific fields if requested
    if fields:
        available = [f for f in fields if f in df.columns]
        if available:
            df = df[available]

    return QueryResult(
        success=True,
        data=df,
    )


def generate_query_plan(query_dsl: dict[str, Any]) -> dict[str, Any]:
    """Generate a human-readable query execution plan.

    Args:
        query_dsl: Query in DSL format

    Returns:
        Dictionary with plan details and Python code
    """
    try:
        validated_query = QueryDSL(**query_dsl)
    except ValidationError as e:
        return {
            "success": False,
            "error": f"Invalid query: {e}",
        }

    plan_steps = []

    # Step 1: Load catalogue
    plan_steps.append("1. Load ATNF catalogue (version detected at runtime)")

    # Step 2: Apply filters
    if validated_query.filters:
        plan_steps.append("2. Apply filters:")
        condition = validated_query.to_psrqpy_condition()
        plan_steps.append(f"   Condition: {condition}")
    else:
        plan_steps.append("2. No filters applied (full catalogue)")

    # Step 3: Select fields
    if validated_query.select_fields:
        fields = ", ".join(validated_query.select_fields)
        plan_steps.append(f"3. Select fields: {fields}")
    else:
        plan_steps.append("3. Select all available fields")

    # Step 4: Order
    if validated_query.order_by:
        direction = "descending" if validated_query.order_desc else "ascending"
        plan_steps.append(f"4. Sort by {validated_query.order_by} ({direction})")
    else:
        plan_steps.append("4. No ordering applied")

    # Step 5: Limit
    if validated_query.limit:
        plan_steps.append(f"5. Limit to {validated_query.limit} results")
    else:
        plan_steps.append("5. No limit (return all matches)")

    return {
        "success": True,
        "plan": "\n".join(plan_steps),
        "condition_string": validated_query.to_psrqpy_condition(),
        "python_code": validated_query.to_python_code(),
        "summary": validated_query.get_query_summary(),
    }
