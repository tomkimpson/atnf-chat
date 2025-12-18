"""Tests for the Query DSL module."""

import pytest
from pydantic import ValidationError

from atnf_chat.core.dsl import (
    ComparisonOp,
    FilterClause,
    FilterGroup,
    LogicalOp,
    QueryDSL,
    make_binary_filter,
    make_gc_filter,
    make_msp_filter,
    make_simple_filter,
)


class TestComparisonOp:
    """Tests for ComparisonOp enum."""

    def test_all_operators_defined(self) -> None:
        """Verify all expected operators are defined."""
        expected = [
            "eq", "ne", "lt", "le", "gt", "ge",
            "contains", "startswith", "in_range",
            "is_null", "not_null"
        ]
        for op in expected:
            assert ComparisonOp(op) is not None

    def test_requires_value(self) -> None:
        """Test which operators require values."""
        # These require values
        for op in [ComparisonOp.EQ, ComparisonOp.LT, ComparisonOp.CONTAINS]:
            assert ComparisonOp.requires_value(op) is True

        # These don't require values
        for op in [ComparisonOp.IS_NULL, ComparisonOp.NOT_NULL]:
            assert ComparisonOp.requires_value(op) is False

    def test_requires_range(self) -> None:
        """Test which operators require range values."""
        assert ComparisonOp.requires_range(ComparisonOp.IN_RANGE) is True
        assert ComparisonOp.requires_range(ComparisonOp.LT) is False
        assert ComparisonOp.requires_range(ComparisonOp.EQ) is False


class TestFilterClause:
    """Tests for FilterClause model."""

    # === Valid Clause Construction ===

    def test_create_simple_clause(self) -> None:
        """Test creating a simple filter clause."""
        clause = FilterClause(field="P0", cmp=ComparisonOp.LT, value=0.03)
        assert clause.field == "P0"
        assert clause.cmp == ComparisonOp.LT
        assert clause.value == 0.03

    def test_create_clause_with_string_operator(self) -> None:
        """Test creating clause with string operator."""
        clause = FilterClause(field="P0", cmp="lt", value=0.03)
        assert clause.cmp == ComparisonOp.LT

    def test_field_normalized_to_uppercase(self) -> None:
        """Test that field names are normalized to uppercase."""
        clause = FilterClause(field="p0", cmp="lt", value=0.03)
        assert clause.field == "P0"

        clause = FilterClause(field="Dm", cmp="gt", value=10.0)
        assert clause.field == "DM"

    def test_field_alias_resolution(self) -> None:
        """Test that field aliases are resolved to canonical codes."""
        clause = FilterClause(field="period", cmp="lt", value=0.03)
        assert clause.field == "P0"

        clause = FilterClause(field="magnetic field", cmp="gt", value=1e10)
        assert clause.field == "BSURF"

    def test_create_string_comparison(self) -> None:
        """Test creating string comparison clause."""
        clause = FilterClause(field="ASSOC", cmp=ComparisonOp.CONTAINS, value="GC")
        assert clause.field == "ASSOC"
        assert clause.value == "GC"

    def test_create_range_clause(self) -> None:
        """Test creating range comparison clause."""
        clause = FilterClause(
            field="P0", cmp=ComparisonOp.IN_RANGE, value=[0.001, 0.01]
        )
        assert clause.value == [0.001, 0.01]

    def test_create_null_check_clause(self) -> None:
        """Test creating null check clauses."""
        clause = FilterClause(field="PB", cmp=ComparisonOp.NOT_NULL)
        assert clause.cmp == ComparisonOp.NOT_NULL
        assert clause.value is None

        clause = FilterClause(field="ECC", cmp=ComparisonOp.IS_NULL)
        assert clause.cmp == ComparisonOp.IS_NULL

    def test_optional_unit(self) -> None:
        """Test that unit is optional and preserved."""
        clause = FilterClause(field="P0", cmp="lt", value=0.03, unit="s")
        assert clause.unit == "s"

    # === Invalid Clause Handling ===

    def test_invalid_field_raises_error(self) -> None:
        """Test that invalid field names raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            FilterClause(field="INVALID_FIELD", cmp="lt", value=1.0)
        assert "Unknown field" in str(exc_info.value)

    def test_missing_value_for_comparison_raises_error(self) -> None:
        """Test that missing value for comparison operator raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FilterClause(field="P0", cmp="lt", value=None)
        assert "requires a value" in str(exc_info.value)

    def test_invalid_range_format_raises_error(self) -> None:
        """Test that invalid range format raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FilterClause(field="P0", cmp="in_range", value=0.03)
        assert "requires a [min, max] range" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            FilterClause(field="P0", cmp="in_range", value=[0.03])
        assert "requires a [min, max] range" in str(exc_info.value)

    def test_invalid_range_order_raises_error(self) -> None:
        """Test that min > max raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FilterClause(field="P0", cmp="in_range", value=[0.1, 0.01])
        assert "min" in str(exc_info.value) and "max" in str(exc_info.value)

    # === Condition String Generation ===

    def test_to_condition_string_numeric_comparison(self) -> None:
        """Test condition string for numeric comparisons."""
        test_cases = [
            (ComparisonOp.LT, 0.03, "(P0 < 0.03)"),
            (ComparisonOp.LE, 0.03, "(P0 <= 0.03)"),
            (ComparisonOp.GT, 0.03, "(P0 > 0.03)"),
            (ComparisonOp.GE, 0.03, "(P0 >= 0.03)"),
            (ComparisonOp.EQ, 0.03, "(P0 == 0.03)"),
            (ComparisonOp.NE, 0.03, "(P0 != 0.03)"),
        ]
        for op, value, expected in test_cases:
            clause = FilterClause(field="P0", cmp=op, value=value)
            assert clause.to_condition_string() == expected

    def test_to_condition_string_contains(self) -> None:
        """Test condition string for contains operator."""
        clause = FilterClause(field="ASSOC", cmp=ComparisonOp.CONTAINS, value="GC")
        assert clause.to_condition_string() == "(ASSOC like '*GC*')"

    def test_to_condition_string_startswith(self) -> None:
        """Test condition string for startswith operator."""
        clause = FilterClause(field="JNAME", cmp=ComparisonOp.STARTSWITH, value="J1")
        assert clause.to_condition_string() == "(JNAME like 'J1*')"

    def test_to_condition_string_in_range(self) -> None:
        """Test condition string for range comparison."""
        clause = FilterClause(field="P0", cmp=ComparisonOp.IN_RANGE, value=[0.001, 0.01])
        assert clause.to_condition_string() == "((P0 >= 0.001) && (P0 <= 0.01))"

    def test_to_condition_string_null_checks(self) -> None:
        """Test condition string for null checks."""
        clause = FilterClause(field="PB", cmp=ComparisonOp.NOT_NULL)
        assert clause.to_condition_string() == "(PB != '')"

        clause = FilterClause(field="PB", cmp=ComparisonOp.IS_NULL)
        assert clause.to_condition_string() == "(PB == '')"


class TestFilterGroup:
    """Tests for FilterGroup model."""

    def test_create_and_group(self) -> None:
        """Test creating AND filter group."""
        group = FilterGroup(
            op=LogicalOp.AND,
            clauses=[
                FilterClause(field="P0", cmp="lt", value=0.03),
                FilterClause(field="DM", cmp="gt", value=10.0),
            ],
        )
        assert group.op == LogicalOp.AND
        assert len(group.clauses) == 2

    def test_create_or_group(self) -> None:
        """Test creating OR filter group."""
        group = FilterGroup(
            op=LogicalOp.OR,
            clauses=[
                FilterClause(field="P0", cmp="lt", value=0.03),
                FilterClause(field="ASSOC", cmp="contains", value="GC"),
            ],
        )
        assert group.op == LogicalOp.OR

    def test_create_with_string_operator(self) -> None:
        """Test creating group with string operator."""
        group = FilterGroup(
            op="and",
            clauses=[FilterClause(field="P0", cmp="lt", value=0.03)],
        )
        assert group.op == LogicalOp.AND

    def test_nested_groups(self) -> None:
        """Test nested filter groups."""
        inner_group = FilterGroup(
            op=LogicalOp.OR,
            clauses=[
                FilterClause(field="ASSOC", cmp="contains", value="GC"),
                FilterClause(field="ASSOC", cmp="contains", value="SNR"),
            ],
        )
        outer_group = FilterGroup(
            op=LogicalOp.AND,
            clauses=[
                FilterClause(field="P0", cmp="lt", value=0.03),
                inner_group,
            ],
        )
        assert len(outer_group.clauses) == 2
        assert isinstance(outer_group.clauses[1], FilterGroup)

    def test_empty_clauses_raises_error(self) -> None:
        """Test that empty clauses list raises error."""
        with pytest.raises(ValidationError):
            FilterGroup(op="and", clauses=[])

    def test_to_condition_string_and(self) -> None:
        """Test AND condition string generation."""
        group = FilterGroup(
            op=LogicalOp.AND,
            clauses=[
                FilterClause(field="P0", cmp="lt", value=0.03),
                FilterClause(field="DM", cmp="gt", value=10.0),
            ],
        )
        expected = "((P0 < 0.03) && (DM > 10.0))"
        assert group.to_condition_string() == expected

    def test_to_condition_string_or(self) -> None:
        """Test OR condition string generation."""
        group = FilterGroup(
            op=LogicalOp.OR,
            clauses=[
                FilterClause(field="P0", cmp="lt", value=0.03),
                FilterClause(field="ASSOC", cmp="contains", value="GC"),
            ],
        )
        expected = "((P0 < 0.03) || (ASSOC like '*GC*'))"
        assert group.to_condition_string() == expected

    def test_to_condition_string_nested(self) -> None:
        """Test nested group condition string."""
        inner = FilterGroup(
            op=LogicalOp.OR,
            clauses=[
                FilterClause(field="ASSOC", cmp="contains", value="GC"),
                FilterClause(field="ASSOC", cmp="contains", value="SNR"),
            ],
        )
        outer = FilterGroup(
            op=LogicalOp.AND,
            clauses=[
                FilterClause(field="P0", cmp="lt", value=0.03),
                inner,
            ],
        )
        result = outer.to_condition_string()
        assert "&&" in result
        assert "||" in result
        assert "P0 < 0.03" in result


class TestQueryDSL:
    """Tests for QueryDSL model."""

    # === Valid Query Construction ===

    def test_create_minimal_query(self) -> None:
        """Test creating query with no filters."""
        query = QueryDSL()
        assert query.select_fields is None
        assert query.filters is None
        assert query.limit is None

    def test_create_query_with_select_fields(self) -> None:
        """Test creating query with field selection."""
        query = QueryDSL(select_fields=["JNAME", "P0", "DM"])
        assert query.select_fields == ["JNAME", "P0", "DM"]

    def test_select_fields_normalized(self) -> None:
        """Test that select fields are normalized."""
        query = QueryDSL(select_fields=["jname", "p0", "dm"])
        assert query.select_fields == ["JNAME", "P0", "DM"]

    def test_select_fields_alias_resolution(self) -> None:
        """Test that select field aliases are resolved."""
        query = QueryDSL(select_fields=["pulsar name", "period", "dispersion measure"])
        assert "JNAME" in query.select_fields
        assert "P0" in query.select_fields
        assert "DM" in query.select_fields

    def test_create_query_with_filters(self) -> None:
        """Test creating query with filters."""
        query = QueryDSL(
            select_fields=["JNAME", "P0"],
            filters=FilterGroup(
                op="and",
                clauses=[FilterClause(field="P0", cmp="lt", value=0.03)],
            ),
        )
        assert query.filters is not None

    def test_create_query_with_order_by(self) -> None:
        """Test creating query with ordering."""
        query = QueryDSL(select_fields=["JNAME", "P0"], order_by="P0")
        assert query.order_by == "P0"

    def test_order_by_normalized(self) -> None:
        """Test that order_by is normalized."""
        query = QueryDSL(order_by="p0")
        assert query.order_by == "P0"

    def test_order_by_alias_resolution(self) -> None:
        """Test that order_by alias is resolved."""
        query = QueryDSL(order_by="period")
        assert query.order_by == "P0"

    def test_create_query_with_limit(self) -> None:
        """Test creating query with limit."""
        query = QueryDSL(limit=100)
        assert query.limit == 100

    def test_create_full_query(self) -> None:
        """Test creating a full query with all options."""
        query = QueryDSL(
            select_fields=["JNAME", "P0", "DM", "ASSOC"],
            filters=FilterGroup(
                op="and",
                clauses=[
                    FilterClause(field="P0", cmp="lt", value=0.03),
                    FilterClause(field="ASSOC", cmp="contains", value="GC"),
                ],
            ),
            order_by="P0",
            order_desc=False,
            limit=100,
        )
        assert query.select_fields == ["JNAME", "P0", "DM", "ASSOC"]
        assert query.filters is not None
        assert query.order_by == "P0"
        assert query.limit == 100

    # === Invalid Query Handling ===

    def test_invalid_select_field_raises_error(self) -> None:
        """Test that invalid select fields raise error."""
        with pytest.raises(ValidationError) as exc_info:
            QueryDSL(select_fields=["JNAME", "INVALID"])
        assert "Unknown field" in str(exc_info.value)

    def test_invalid_order_by_raises_error(self) -> None:
        """Test that invalid order_by raises error."""
        with pytest.raises(ValidationError) as exc_info:
            QueryDSL(order_by="INVALID")
        assert "Unknown field" in str(exc_info.value)

    def test_limit_too_small_raises_error(self) -> None:
        """Test that limit < 1 raises error."""
        with pytest.raises(ValidationError):
            QueryDSL(limit=0)

    def test_limit_too_large_raises_error(self) -> None:
        """Test that limit > 10000 raises error."""
        with pytest.raises(ValidationError):
            QueryDSL(limit=10001)

    # === Condition String Generation ===

    def test_to_psrqpy_condition_no_filters(self) -> None:
        """Test condition string with no filters."""
        query = QueryDSL()
        assert query.to_psrqpy_condition() is None

    def test_to_psrqpy_condition_simple(self) -> None:
        """Test simple condition string."""
        query = QueryDSL(
            filters=FilterGroup(
                op="and",
                clauses=[FilterClause(field="P0", cmp="lt", value=0.03)],
            )
        )
        assert query.to_psrqpy_condition() == "((P0 < 0.03))"

    def test_to_psrqpy_condition_complex(self) -> None:
        """Test complex condition string."""
        query = QueryDSL(
            filters=FilterGroup(
                op="and",
                clauses=[
                    FilterClause(field="P0", cmp="lt", value=0.03),
                    FilterClause(field="ASSOC", cmp="contains", value="GC"),
                ],
            )
        )
        condition = query.to_psrqpy_condition()
        assert "P0 < 0.03" in condition
        assert "ASSOC like '*GC*'" in condition
        assert "&&" in condition

    def test_to_psrqpy_params(self) -> None:
        """Test parameter list generation."""
        query = QueryDSL(select_fields=["JNAME", "P0", "DM"])
        assert query.to_psrqpy_params() == ["JNAME", "P0", "DM"]

        query = QueryDSL()
        assert query.to_psrqpy_params() is None

    # === Query Summary ===

    def test_get_query_summary(self) -> None:
        """Test query summary generation."""
        query = QueryDSL(
            select_fields=["JNAME", "P0"],
            filters=FilterGroup(
                op="and",
                clauses=[
                    FilterClause(field="P0", cmp="lt", value=0.03),
                    FilterClause(field="DM", cmp="gt", value=10.0),
                ],
            ),
            order_by="P0",
            limit=100,
        )
        summary = query.get_query_summary()
        assert summary["select"] == ["JNAME", "P0"]
        assert summary["filter_count"] == 2
        assert "P0 ASC" in summary["order_by"]
        assert summary["limit"] == 100

    def test_get_query_summary_minimal(self) -> None:
        """Test summary for minimal query."""
        query = QueryDSL()
        summary = query.get_query_summary()
        assert summary["select"] == "all fields"
        assert summary["condition"] is None
        assert summary["filter_count"] == 0

    # === Python Code Generation ===

    def test_to_python_code_minimal(self) -> None:
        """Test Python code generation for minimal query."""
        query = QueryDSL()
        code = query.to_python_code()
        assert "from psrqpy import QueryATNF" in code
        assert "query = QueryATNF()" in code
        assert "results = query.pandas" in code

    def test_to_python_code_with_params(self) -> None:
        """Test Python code with select fields."""
        query = QueryDSL(select_fields=["JNAME", "P0"])
        code = query.to_python_code()
        assert "params=" in code
        assert "JNAME" in code
        assert "P0" in code

    def test_to_python_code_with_condition(self) -> None:
        """Test Python code with condition."""
        query = QueryDSL(
            filters=FilterGroup(
                op="and",
                clauses=[FilterClause(field="P0", cmp="lt", value=0.03)],
            )
        )
        code = query.to_python_code()
        assert "condition=" in code

    def test_to_python_code_with_ordering(self) -> None:
        """Test Python code with ordering."""
        query = QueryDSL(order_by="P0", order_desc=True)
        code = query.to_python_code()
        assert "sort_values" in code
        assert "P0" in code
        assert "ascending=False" in code

    def test_to_python_code_with_limit(self) -> None:
        """Test Python code with limit."""
        query = QueryDSL(limit=100)
        code = query.to_python_code()
        assert ".head(100)" in code


class TestConvenienceFunctions:
    """Tests for convenience filter functions."""

    def test_make_simple_filter(self) -> None:
        """Test simple filter creation."""
        group = make_simple_filter("P0", "lt", 0.03)
        assert isinstance(group, FilterGroup)
        assert len(group.clauses) == 1
        assert group.clauses[0].field == "P0"

    def test_make_msp_filter(self) -> None:
        """Test MSP filter creation."""
        group = make_msp_filter()
        condition = group.to_condition_string()
        assert "P0 < 0.03" in condition

        group = make_msp_filter(max_period=0.01)
        condition = group.to_condition_string()
        assert "P0 < 0.01" in condition

    def test_make_binary_filter(self) -> None:
        """Test binary filter creation."""
        group = make_binary_filter()
        assert len(group.clauses) == 1
        assert group.clauses[0].field == "BINARY"
        assert group.clauses[0].cmp == ComparisonOp.NOT_NULL

    def test_make_gc_filter(self) -> None:
        """Test globular cluster filter creation."""
        group = make_gc_filter()
        condition = group.to_condition_string()
        assert "ASSOC" in condition
        assert "GC" in condition


class TestDSLFromDict:
    """Tests for creating DSL objects from dictionaries (LLM output)."""

    def test_query_from_dict(self) -> None:
        """Test creating QueryDSL from dictionary."""
        data = {
            "select_fields": ["JNAME", "P0", "DM"],
            "filters": {
                "op": "and",
                "clauses": [
                    {"field": "P0", "cmp": "lt", "value": 0.03},
                    {"field": "ASSOC", "cmp": "contains", "value": "GC"},
                ],
            },
            "order_by": "P0",
            "limit": 100,
        }
        query = QueryDSL(**data)
        assert query.select_fields == ["JNAME", "P0", "DM"]
        assert query.filters is not None
        assert len(query.filters.clauses) == 2

    def test_nested_filter_from_dict(self) -> None:
        """Test creating nested filters from dictionary."""
        data = {
            "filters": {
                "op": "and",
                "clauses": [
                    {"field": "P0", "cmp": "lt", "value": 0.03},
                    {
                        "op": "or",
                        "clauses": [
                            {"field": "ASSOC", "cmp": "contains", "value": "GC"},
                            {"field": "ASSOC", "cmp": "contains", "value": "SNR"},
                        ],
                    },
                ],
            }
        }
        query = QueryDSL(**data)
        assert query.filters is not None
        assert len(query.filters.clauses) == 2
        assert isinstance(query.filters.clauses[1], FilterGroup)
