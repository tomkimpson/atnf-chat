"""Core functionality for ATNF-Chat.

This module contains:
- Query DSL definition and validation
- Catalogue interface with psrqpy
- Schema grounding pack for parameter mappings
- Result validation and provenance tracking
"""

from atnf_chat.core.schema import (
    ParameterCategory,
    ParameterDefinition,
    ParameterType,
    SchemaGroundingPack,
)

__all__ = [
    "ParameterCategory",
    "ParameterDefinition",
    "ParameterType",
    "SchemaGroundingPack",
]


def __getattr__(name: str):
    """Lazy imports for modules not yet implemented."""
    if name == "CatalogueInterface":
        from atnf_chat.core.catalogue import CatalogueInterface

        return CatalogueInterface
    if name in ("ComparisonOp", "FilterClause", "FilterGroup", "LogicalOp", "QueryDSL"):
        from atnf_chat.core import dsl

        return getattr(dsl, name)
    if name == "ResultValidator":
        from atnf_chat.core.validation import ResultValidator

        return ResultValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
