"""Pytest configuration and fixtures for ATNF-Chat tests."""

import pytest


@pytest.fixture
def sample_query_dsl() -> dict:
    """Return a sample valid query DSL for testing."""
    return {
        "select_fields": ["JNAME", "P0", "DM"],
        "filters": {
            "op": "and",
            "clauses": [
                {"field": "P0", "cmp": "lt", "value": 0.03, "unit": "s"},
            ],
        },
        "order_by": "P0",
        "limit": 100,
    }


@pytest.fixture
def sample_msp_query_dsl() -> dict:
    """Return a query DSL for millisecond pulsars in globular clusters."""
    return {
        "select_fields": ["JNAME", "P0", "DM", "ASSOC"],
        "filters": {
            "op": "and",
            "clauses": [
                {"field": "P0", "cmp": "lt", "value": 0.03, "unit": "s"},
                {"field": "ASSOC", "cmp": "contains", "value": "GC"},
            ],
        },
        "order_by": "P0",
        "limit": 100,
    }


@pytest.fixture
def sample_invalid_query_dsl() -> dict:
    """Return an invalid query DSL for testing error handling."""
    return {
        "select_fields": ["INVALID_FIELD"],
        "filters": None,
    }


@pytest.fixture
def mock_catalogue_metadata() -> dict:
    """Return mock catalogue metadata for testing."""
    return {
        "version": "2.0.0",
        "snapshot_date": "2025-01-15T10:30:00",
        "total_pulsars": 3500,
        "measured_parameters": {
            "JNAME": 3500,
            "P0": 3450,
            "DM": 3400,
            "BSURF": 2800,
        },
    }
