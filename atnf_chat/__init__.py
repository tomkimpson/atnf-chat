"""ATNF-Chat: LLM-Powered Conversational Interface for Pulsar Catalogue Queries.

This package provides a natural language interface to the ATNF Pulsar Catalogue,
enabling researchers to query pulsar data, generate visualizations, and perform
analyses through conversational interactions.
"""

__version__ = "0.1.0"
__author__ = "Tom Kimpson"


def __getattr__(name: str):
    """Lazy imports for main package exports."""
    if name == "SchemaGroundingPack":
        from atnf_chat.core.schema import SchemaGroundingPack

        return SchemaGroundingPack
    if name == "QueryDSL":
        from atnf_chat.core.dsl import QueryDSL

        return QueryDSL
    if name == "CatalogueInterface":
        from atnf_chat.core.catalogue import CatalogueInterface

        return CatalogueInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CatalogueInterface",
    "QueryDSL",
    "SchemaGroundingPack",
    "__version__",
]
