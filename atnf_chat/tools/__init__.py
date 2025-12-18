"""LLM function tools for ATNF-Chat.

This module contains the tools exposed to the LLM for function calling:
- query_catalogue: Execute validated DSL queries
- compute_derived_parameter: Calculate physics quantities
- statistical_analysis: Compute summary statistics
- correlation_analysis: Analyze relationships between parameters
- generate_query_plan: Show execution plan and reproducible code
"""

from atnf_chat.tools.analysis import (
    AnalysisResult,
    CorrelationResult,
    StatisticalSummary,
    compare_groups,
    correlation_analysis,
    multi_correlation_analysis,
    statistical_analysis,
)
from atnf_chat.tools.definitions import (
    ALL_TOOLS,
    get_tool_names,
    get_tools_for_claude,
)
from atnf_chat.tools.derived import (
    DerivedParameterResult,
    compute_derived_parameter,
    get_available_derived_parameters,
)
from atnf_chat.tools.query import (
    QueryResult,
    generate_query_plan,
    get_pulsar_info,
    query_catalogue,
)

__all__ = [
    # Query tools
    "query_catalogue",
    "get_pulsar_info",
    "generate_query_plan",
    "QueryResult",
    # Derived parameters
    "compute_derived_parameter",
    "get_available_derived_parameters",
    "DerivedParameterResult",
    # Analysis tools
    "statistical_analysis",
    "correlation_analysis",
    "multi_correlation_analysis",
    "compare_groups",
    "StatisticalSummary",
    "CorrelationResult",
    "AnalysisResult",
    # Tool definitions
    "get_tools_for_claude",
    "get_tool_names",
    "ALL_TOOLS",
]
