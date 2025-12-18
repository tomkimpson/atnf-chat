"""LLM tool definitions for Anthropic Claude function calling.

This module defines the tool schemas that are passed to the Claude API
for function calling. Each tool definition includes:
- Name and description
- Input schema (JSON Schema format)
- Examples of usage

These definitions follow the Anthropic tool use specification.
"""

from typing import Any

# Query Catalogue Tool
QUERY_CATALOGUE_TOOL: dict[str, Any] = {
    "name": "query_catalogue",
    "description": """Query the ATNF Pulsar Catalogue using a validated DSL (Domain Specific Language).

Use this tool to search for pulsars based on their properties. The query uses a structured format
that is validated before execution to prevent errors.

IMPORTANT: Always use the DSL format, never free-form filter strings.

Returns:
- Matching pulsars with requested fields
- Result count and data quality information
- Null counts for each field (important for derived calculations)
- Catalogue version for reproducibility""",
    "input_schema": {
        "type": "object",
        "properties": {
            "select_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of ATNF parameter codes to return. Common fields: JNAME, P0, P1, F0, DM, BSURF, EDOT, AGE, PB, ECC, ASSOC. If omitted, returns all fields.",
            },
            "filters": {
                "type": "object",
                "description": "Filter conditions in DSL format",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["and", "or"],
                        "description": "Logical operator to combine clauses",
                    },
                    "clauses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "description": "ATNF parameter code (e.g., P0, DM, ASSOC)",
                                },
                                "cmp": {
                                    "type": "string",
                                    "enum": [
                                        "eq",
                                        "ne",
                                        "lt",
                                        "le",
                                        "gt",
                                        "ge",
                                        "contains",
                                        "startswith",
                                        "in_range",
                                        "is_null",
                                        "not_null",
                                    ],
                                    "description": "Comparison operator",
                                },
                                "value": {
                                    "description": "Comparison value. Use number for numeric comparisons, string for text, [min, max] array for in_range, or omit for is_null/not_null",
                                },
                                "unit": {
                                    "type": "string",
                                    "description": "Optional unit for documentation (e.g., 's' for seconds)",
                                },
                            },
                            "required": ["field", "cmp"],
                        },
                        "description": "List of filter clauses",
                    },
                },
                "required": ["op", "clauses"],
            },
            "order_by": {
                "type": "string",
                "description": "Field to sort results by",
            },
            "order_desc": {
                "type": "boolean",
                "description": "Sort descending if true (default: false = ascending)",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "description": "Maximum number of results to return",
            },
        },
    },
}

# Get Pulsar Info Tool
GET_PULSAR_INFO_TOOL: dict[str, Any] = {
    "name": "get_pulsar_info",
    "description": """Get detailed information about a specific pulsar by name.

Use this tool when the user asks about a specific pulsar (e.g., "Tell me about the Vela pulsar").
Searches by JNAME (J2000 name) or BNAME (B1950 name), with partial matching support.

Returns all available parameters for the pulsar.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Pulsar name (e.g., 'J0835-4510' for Vela, 'J0437-4715', 'B1919+21')",
            },
        },
        "required": ["name"],
    },
}

# Compute Derived Parameter Tool
COMPUTE_DERIVED_TOOL: dict[str, Any] = {
    "name": "compute_derived_parameter",
    "description": """Compute a derived physical parameter for pulsars.

IMPORTANT: This tool prefers ATNF-native values when available. If the catalogue already
contains the derived parameter (like BSURF, EDOT, AGE), those values are used.
Only when necessary are values computed from basic parameters, with all assumptions documented.

Available derived parameters:
- BSURF: Surface magnetic field (Gauss)
- EDOT: Spin-down luminosity (erg/s)
- AGE: Characteristic age (years)
- B_LC: Magnetic field at light cylinder (Gauss)
- VTRANS: Transverse velocity (km/s)

Returns:
- Computed values
- Source (atnf_native or computed)
- Formula and assumptions used
- Completeness (fraction with valid values)""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_dsl": {
                "type": "object",
                "description": "Query DSL to select pulsars for computation. Use the same format as query_catalogue (select_fields, filters, order_by, limit).",
            },
            "parameter": {
                "type": "string",
                "enum": ["BSURF", "EDOT", "AGE", "B_LC", "VTRANS"],
                "description": "Name of derived parameter to compute",
            },
            "use_atnf_native": {
                "type": "boolean",
                "description": "If true (default), prefer ATNF-provided values when available",
            },
        },
        "required": ["parameter"],
    },
}

# Statistical Analysis Tool
STATISTICAL_ANALYSIS_TOOL: dict[str, Any] = {
    "name": "statistical_analysis",
    "description": """Compute summary statistics for pulsar parameters.

Use this tool to get statistical summaries including:
- Count, mean, median, standard deviation
- Min, max, quartiles
- Skewness and kurtosis

Good for exploring distributions and understanding data quality.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_dsl": {
                "type": "object",
                "description": "Query DSL to select pulsars for analysis. Use the same format as query_catalogue (select_fields, filters, order_by, limit).",
            },
            "parameters": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of parameter names to analyze. If omitted, analyzes all numeric columns.",
            },
        },
    },
}

# Correlation Analysis Tool
CORRELATION_ANALYSIS_TOOL: dict[str, Any] = {
    "name": "correlation_analysis",
    "description": """Analyze correlation between two pulsar parameters.

Computes both Pearson (linear) and Spearman (monotonic) correlation coefficients
with significance tests.

Use use_log=true for parameters with power-law relationships (like P vs Pdot, P vs BSURF).""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_dsl": {
                "type": "object",
                "description": "Query DSL to select pulsars for analysis. Use the same format as query_catalogue (select_fields, filters, order_by, limit).",
            },
            "param_x": {
                "type": "string",
                "description": "First parameter name",
            },
            "param_y": {
                "type": "string",
                "description": "Second parameter name",
            },
            "use_log": {
                "type": "boolean",
                "description": "If true, use log10 of values (for power-law relationships)",
            },
        },
        "required": ["param_x", "param_y"],
    },
}

# Generate Query Plan Tool
GENERATE_QUERY_PLAN_TOOL: dict[str, Any] = {
    "name": "generate_query_plan",
    "description": """Generate a human-readable execution plan for a query.

Shows:
- Step-by-step query execution
- The psrqpy condition string
- Reproducible Python code

Use this when the user asks to "show the query" or wants to understand how a query works.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_dsl": {
                "type": "object",
                "description": "Query in DSL format",
            },
        },
        "required": ["query_dsl"],
    },
}

# All tools for export
ALL_TOOLS: list[dict[str, Any]] = [
    QUERY_CATALOGUE_TOOL,
    GET_PULSAR_INFO_TOOL,
    COMPUTE_DERIVED_TOOL,
    STATISTICAL_ANALYSIS_TOOL,
    CORRELATION_ANALYSIS_TOOL,
    GENERATE_QUERY_PLAN_TOOL,
]


def get_tools_for_claude() -> list[dict[str, Any]]:
    """Get all tool definitions formatted for Claude API.

    Returns:
        List of tool definitions
    """
    return ALL_TOOLS


def get_tool_names() -> list[str]:
    """Get list of all tool names.

    Returns:
        List of tool name strings
    """
    return [tool["name"] for tool in ALL_TOOLS]
