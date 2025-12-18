"""Query DSL (Domain Specific Language) for ATNF Pulsar Catalogue.

This module provides a validated, type-safe DSL for constructing catalogue queries.
The DSL is designed to:
- Prevent LLM drift through strict validation
- Enable pre-execution error checking
- Support composable, nested filter conditions
- Convert to psrqpy condition strings

Example:
    >>> from atnf_chat.core.dsl import QueryDSL, FilterGroup, FilterClause, ComparisonOp
    >>> query = QueryDSL(
    ...     select_fields=["JNAME", "P0", "DM"],
    ...     filters=FilterGroup(
    ...         op="and",
    ...         clauses=[
    ...             FilterClause(field="P0", cmp="lt", value=0.03),
    ...             FilterClause(field="ASSOC", cmp="contains", value="GC")
    ...         ]
    ...     ),
    ...     order_by="P0",
    ...     limit=100
    ... )
    >>> query.to_psrqpy_condition()
    '((P0 < 0.03) && (ASSOC like \\'*GC*\\'))'
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ComparisonOp(str, Enum):
    """Comparison operators for filter clauses.

    These operators define how field values are compared in queries.
    """

    EQ = "eq"  # equals
    NE = "ne"  # not equals
    LT = "lt"  # less than
    LE = "le"  # less than or equal
    GT = "gt"  # greater than
    GE = "ge"  # greater than or equal
    CONTAINS = "contains"  # string contains (case-insensitive)
    STARTSWITH = "startswith"  # string starts with
    IN_RANGE = "in_range"  # value in [min, max] inclusive
    IS_NULL = "is_null"  # field is null/missing
    NOT_NULL = "not_null"  # field is measured (not null)

    @classmethod
    def requires_value(cls, op: ComparisonOp) -> bool:
        """Check if operator requires a comparison value."""
        return op not in (cls.IS_NULL, cls.NOT_NULL)

    @classmethod
    def requires_range(cls, op: ComparisonOp) -> bool:
        """Check if operator requires a range value [min, max]."""
        return op == cls.IN_RANGE


class LogicalOp(str, Enum):
    """Logical operators for combining filter clauses."""

    AND = "and"
    OR = "or"


class FilterClause(BaseModel):
    """A single filter condition in a query.

    Attributes:
        field: ATNF parameter code (e.g., "P0", "DM", "ASSOC")
        cmp: Comparison operator
        value: Comparison value (type depends on operator)
        unit: Optional unit for the value (for documentation/validation)

    Example:
        >>> clause = FilterClause(field="P0", cmp="lt", value=0.03, unit="s")
        >>> clause.to_condition_string()
        '(P0 < 0.03)'
    """

    model_config = ConfigDict(use_enum_values=False, validate_assignment=True)

    field: str = Field(..., description="ATNF parameter code")
    cmp: ComparisonOp = Field(..., description="Comparison operator")
    value: float | str | list[float] | None = Field(
        default=None, description="Comparison value"
    )
    unit: str | None = Field(default=None, description="Unit for the value")

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validate and normalize field name to uppercase."""
        from atnf_chat.core.schema import SchemaGroundingPack

        schema = SchemaGroundingPack()
        normalized = v.upper()

        # Check if it's a valid parameter
        if not schema.is_valid_parameter(normalized):
            # Try to resolve as alias
            resolved = schema.resolve_alias(v)
            if resolved:
                return resolved
            # Get valid fields for error message
            valid_fields = schema.get_all_codes()
            raise ValueError(
                f"Unknown field: '{v}'. Must be a valid ATNF parameter code. "
                f"Common fields: JNAME, P0, P1, F0, DM, BSURF, EDOT, AGE, PB, ECC, ASSOC"
            )
        return normalized

    @model_validator(mode="after")
    def validate_value_for_operator(self) -> Self:
        """Validate that value is appropriate for the operator."""
        if ComparisonOp.requires_value(self.cmp):
            if self.value is None:
                raise ValueError(
                    f"Operator '{self.cmp.value}' requires a value, but none provided"
                )
            if ComparisonOp.requires_range(self.cmp):
                if not isinstance(self.value, list) or len(self.value) != 2:
                    raise ValueError(
                        f"Operator '{self.cmp.value}' requires a [min, max] range, "
                        f"got: {self.value}"
                    )
                if self.value[0] > self.value[1]:
                    raise ValueError(
                        f"Invalid range: min ({self.value[0]}) > max ({self.value[1]})"
                    )
        else:
            # IS_NULL and NOT_NULL should not have a value
            if self.value is not None:
                # Silently ignore value for null checks (be lenient)
                pass
        return self

    def to_condition_string(self) -> str:
        """Convert clause to psrqpy condition string.

        Returns:
            String condition for psrqpy query
        """
        field = self.field
        op = self.cmp

        # Null checks
        if op == ComparisonOp.IS_NULL:
            # psrqpy doesn't have direct null check, use existence
            return f"({field} == '')"
        if op == ComparisonOp.NOT_NULL:
            return f"({field} != '')"

        # Comparison operators
        op_map = {
            ComparisonOp.EQ: "==",
            ComparisonOp.NE: "!=",
            ComparisonOp.LT: "<",
            ComparisonOp.LE: "<=",
            ComparisonOp.GT: ">",
            ComparisonOp.GE: ">=",
        }

        if op in op_map:
            if isinstance(self.value, str):
                return f"({field} {op_map[op]} '{self.value}')"
            return f"({field} {op_map[op]} {self.value})"

        if op == ComparisonOp.CONTAINS:
            # psrqpy uses 'like' with wildcards
            return f"({field} like '*{self.value}*')"

        if op == ComparisonOp.STARTSWITH:
            return f"({field} like '{self.value}*')"

        if op == ComparisonOp.IN_RANGE:
            min_val, max_val = self.value  # type: ignore
            return f"(({field} >= {min_val}) && ({field} <= {max_val}))"

        raise ValueError(f"Unknown operator: {op}")


class FilterGroup(BaseModel):
    """A group of filter conditions combined with a logical operator.

    Supports nested groups for complex queries like:
    (A AND B) OR (C AND D)

    Attributes:
        op: Logical operator (and/or) to combine clauses
        clauses: List of FilterClause or nested FilterGroup

    Example:
        >>> group = FilterGroup(
        ...     op="and",
        ...     clauses=[
        ...         FilterClause(field="P0", cmp="lt", value=0.03),
        ...         FilterClause(field="DM", cmp="gt", value=10.0)
        ...     ]
        ... )
        >>> group.to_condition_string()
        '((P0 < 0.03) && (DM > 10.0))'
    """

    model_config = ConfigDict(use_enum_values=False, validate_assignment=True)

    op: LogicalOp = Field(..., description="Logical operator")
    clauses: list[FilterClause | FilterGroup] = Field(
        ..., min_length=1, description="Filter clauses or nested groups"
    )

    @field_validator("clauses")
    @classmethod
    def validate_clauses_not_empty(
        cls, v: list[FilterClause | FilterGroup]
    ) -> list[FilterClause | FilterGroup]:
        """Ensure at least one clause is provided."""
        if not v:
            raise ValueError("FilterGroup must have at least one clause")
        return v

    def to_condition_string(self) -> str:
        """Convert filter group to psrqpy condition string.

        Returns:
            String condition with proper logical operators
        """
        op_str = " && " if self.op == LogicalOp.AND else " || "
        subconditions = [clause.to_condition_string() for clause in self.clauses]
        return f"({op_str.join(subconditions)})"


class QueryDSL(BaseModel):
    """Complete query definition for ATNF Pulsar Catalogue.

    This is the main interface for constructing validated catalogue queries.
    The DSL ensures all queries are well-formed before execution.

    Attributes:
        select_fields: List of ATNF parameter codes to return (None = all)
        filters: Optional filter conditions
        order_by: Optional field to sort results by
        order_desc: Sort descending if True (default: False = ascending)
        limit: Maximum number of results (1-10000)

    Example:
        >>> query = QueryDSL(
        ...     select_fields=["JNAME", "P0", "DM", "ASSOC"],
        ...     filters=FilterGroup(
        ...         op="and",
        ...         clauses=[
        ...             FilterClause(field="P0", cmp="lt", value=0.03),
        ...             FilterClause(field="ASSOC", cmp="contains", value="GC")
        ...         ]
        ...     ),
        ...     order_by="P0",
        ...     limit=100
        ... )
        >>> print(query.to_psrqpy_condition())
        ((P0 < 0.03) && (ASSOC like '*GC*'))
    """

    model_config = ConfigDict(use_enum_values=False, validate_assignment=True)

    select_fields: list[str] | None = Field(
        default=None, description="Fields to return (None = all available)"
    )
    filters: FilterGroup | None = Field(
        default=None, description="Filter conditions"
    )
    order_by: str | None = Field(default=None, description="Field to sort by")
    order_desc: bool = Field(default=False, description="Sort descending")
    limit: Annotated[int | None, Field(ge=1, le=10000)] = Field(
        default=None, description="Maximum results (1-10000)"
    )

    @field_validator("select_fields")
    @classmethod
    def validate_select_fields(cls, v: list[str] | None) -> list[str] | None:
        """Validate all select fields are valid ATNF parameters."""
        if v is None:
            return v

        from atnf_chat.core.schema import SchemaGroundingPack

        schema = SchemaGroundingPack()
        validated = []

        for field in v:
            normalized = field.upper()
            if schema.is_valid_parameter(normalized):
                validated.append(normalized)
            else:
                # Try alias resolution
                resolved = schema.resolve_alias(field)
                if resolved:
                    validated.append(resolved)
                else:
                    raise ValueError(
                        f"Unknown field in select_fields: '{field}'. "
                        f"Must be a valid ATNF parameter code."
                    )

        return validated

    @field_validator("order_by")
    @classmethod
    def validate_order_by(cls, v: str | None) -> str | None:
        """Validate order_by is a valid ATNF parameter."""
        if v is None:
            return v

        from atnf_chat.core.schema import SchemaGroundingPack

        schema = SchemaGroundingPack()
        normalized = v.upper()

        if schema.is_valid_parameter(normalized):
            return normalized

        # Try alias resolution
        resolved = schema.resolve_alias(v)
        if resolved:
            return resolved

        raise ValueError(
            f"Unknown field for order_by: '{v}'. "
            f"Must be a valid ATNF parameter code."
        )

    def to_psrqpy_condition(self) -> str | None:
        """Convert DSL filters to psrqpy condition string.

        Returns:
            Condition string for psrqpy, or None if no filters
        """
        if self.filters is None:
            return None
        return self.filters.to_condition_string()

    def to_psrqpy_params(self) -> list[str] | None:
        """Get parameter list for psrqpy query.

        Returns:
            List of parameter codes, or None for all parameters
        """
        return self.select_fields

    def get_query_summary(self) -> dict[str, Any]:
        """Get a human-readable summary of the query.

        Returns:
            Dictionary with query components
        """
        summary: dict[str, Any] = {}

        if self.select_fields:
            summary["select"] = self.select_fields
        else:
            summary["select"] = "all fields"

        if self.filters:
            summary["condition"] = self.to_psrqpy_condition()
            summary["filter_count"] = self._count_filters(self.filters)
        else:
            summary["condition"] = None
            summary["filter_count"] = 0

        if self.order_by:
            direction = "DESC" if self.order_desc else "ASC"
            summary["order_by"] = f"{self.order_by} {direction}"

        if self.limit:
            summary["limit"] = self.limit

        return summary

    def _count_filters(self, group: FilterGroup) -> int:
        """Count total number of filter clauses."""
        count = 0
        for clause in group.clauses:
            if isinstance(clause, FilterClause):
                count += 1
            else:
                count += self._count_filters(clause)
        return count

    def to_python_code(self) -> str:
        """Generate reproducible Python code for this query.

        Returns:
            Python code string that reproduces this query
        """
        lines = [
            "from psrqpy import QueryATNF",
            "",
        ]

        params = self.to_psrqpy_params()
        condition = self.to_psrqpy_condition()

        # Build query call
        args = []
        if params:
            args.append(f"params={params}")
        if condition:
            # Escape quotes in condition
            escaped = condition.replace("'", "\\'")
            args.append(f"condition='{escaped}'")

        if args:
            lines.append(f"query = QueryATNF({', '.join(args)})")
        else:
            lines.append("query = QueryATNF()")

        lines.append("results = query.pandas")

        # Add sorting if specified
        if self.order_by:
            ascending = "True" if not self.order_desc else "False"
            lines.append(
                f"results = results.sort_values('{self.order_by}', ascending={ascending})"
            )

        # Add limit if specified
        if self.limit:
            lines.append(f"results = results.head({self.limit})")

        return "\n".join(lines)


# Convenience functions for building queries


def make_simple_filter(field: str, cmp: str, value: Any) -> FilterGroup:
    """Create a simple single-clause filter.

    Args:
        field: ATNF parameter code
        cmp: Comparison operator string
        value: Comparison value

    Returns:
        FilterGroup with single clause
    """
    return FilterGroup(
        op=LogicalOp.AND,
        clauses=[FilterClause(field=field, cmp=ComparisonOp(cmp), value=value)],
    )


def make_msp_filter(max_period: float = 0.03) -> FilterGroup:
    """Create a filter for millisecond pulsars.

    Args:
        max_period: Maximum period in seconds (default: 30ms)

    Returns:
        FilterGroup for MSP selection
    """
    return FilterGroup(
        op=LogicalOp.AND,
        clauses=[FilterClause(field="P0", cmp=ComparisonOp.LT, value=max_period)],
    )


def make_binary_filter() -> FilterGroup:
    """Create a filter for binary pulsars.

    Returns:
        FilterGroup selecting pulsars with binary companions
    """
    return FilterGroup(
        op=LogicalOp.AND,
        clauses=[FilterClause(field="BINARY", cmp=ComparisonOp.NOT_NULL)],
    )


def make_gc_filter() -> FilterGroup:
    """Create a filter for pulsars in globular clusters.

    Returns:
        FilterGroup for globular cluster pulsars
    """
    return FilterGroup(
        op=LogicalOp.AND,
        clauses=[FilterClause(field="ASSOC", cmp=ComparisonOp.CONTAINS, value="GC")],
    )
