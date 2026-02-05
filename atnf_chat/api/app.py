"""FastAPI application for ATNF-Chat.

This module provides the REST API endpoints for the ATNF-Chat application,
including chat interface, query execution, and visualization generation.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from atnf_chat.config import get_settings
from atnf_chat.core.catalogue import CatalogueInterface, get_catalogue
from atnf_chat.core.dsl import QueryDSL
from atnf_chat.core.validation import ResultValidator
from atnf_chat.tools import (
    compute_derived_parameter,
    correlation_analysis,
    generate_query_plan,
    get_pulsar_info,
    query_catalogue,
    statistical_analysis,
)
from atnf_chat.visualization import (
    create_comparison_plot,
    create_histogram,
    create_pp_diagram,
    create_scatter_plot,
    create_sky_plot,
)
from atnf_chat.api.chat import router as chat_router
from atnf_chat.api.rate_limit import RateLimitMiddleware, get_rate_limiter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


# Global state
_catalogue: CatalogueInterface | None = None
_validator: ResultValidator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown."""
    global _catalogue, _validator

    logger.info("Starting ATNF-Chat API")
    settings = get_settings()

    # Initialize catalogue (lazy load on first request if needed)
    try:
        _catalogue = get_catalogue()
        logger.info(f"Catalogue loaded: version {_catalogue.version}")
    except Exception as e:
        logger.warning(f"Could not preload catalogue: {e}")
        _catalogue = None

    _validator = ResultValidator()

    yield

    logger.info("Shutting down ATNF-Chat API")


# Create FastAPI app
app = FastAPI(
    title="ATNF-Chat API",
    description="Natural language interface to the ATNF Pulsar Catalogue",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware (protects /chat/ when using server API key)
app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=get_rate_limiter(),
    protected_paths=["/chat/"],
)

# Include routers
app.include_router(chat_router)


# Request/Response models
class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    messages: list[ChatMessage] = Field(..., description="Conversation history")
    stream: bool = Field(default=True, description="Whether to stream response")


class QueryRequest(BaseModel):
    """Request body for query endpoint."""

    query_dsl: dict[str, Any] = Field(..., description="Query DSL specification")


class QueryResponse(BaseModel):
    """Response from query endpoint."""

    success: bool
    data: list[dict[str, Any]] | None = None
    result_count: int | None = None
    error: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] | None = None


class PulsarInfoRequest(BaseModel):
    """Request for pulsar information."""

    pulsar_name: str = Field(..., description="Pulsar name (JNAME or BNAME)")
    fields: list[str] | None = Field(default=None, description="Specific fields to return")


class DerivedParameterRequest(BaseModel):
    """Request for derived parameter computation."""

    query_dsl: dict[str, Any] = Field(..., description="Query to get pulsar data")
    parameter: str = Field(..., description="Derived parameter name")
    use_atnf_native: bool = Field(default=True, description="Prefer ATNF-native values")


class AnalysisRequest(BaseModel):
    """Request for statistical analysis."""

    query_dsl: dict[str, Any] = Field(..., description="Query to get pulsar data")
    parameters: list[str] | None = Field(default=None, description="Parameters to analyze")


class CorrelationRequest(BaseModel):
    """Request for correlation analysis."""

    query_dsl: dict[str, Any] = Field(..., description="Query to get pulsar data")
    param_x: str = Field(..., description="X-axis parameter")
    param_y: str = Field(..., description="Y-axis parameter")
    use_log: bool = Field(default=False, description="Use log transform")


class PlotRequest(BaseModel):
    """Request for plot generation."""

    plot_type: str = Field(..., description="Type of plot to create")
    query_dsl: dict[str, Any] = Field(..., description="Query to get pulsar data")
    options: dict[str, Any] = Field(default_factory=dict, description="Plot options")


class PlotResponse(BaseModel):
    """Response with plot data."""

    success: bool
    plot_json: str | None = None
    plot_html: str | None = None
    error: str | None = None


class CodeExportRequest(BaseModel):
    """Request for code export."""

    query_dsl: dict[str, Any] = Field(..., description="Query DSL to export")
    include_analysis: bool = Field(default=False, description="Include analysis code")


class CodeExportResponse(BaseModel):
    """Response with exported code."""

    python_code: str
    description: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    catalogue_version: str | None = None
    pulsar_count: int | None = None


class ApiStatusResponse(BaseModel):
    """API status response including key availability."""

    has_server_key: bool = Field(
        description="Whether the server has an API key configured"
    )
    rate_limit_per_minute: int = Field(
        description="Rate limit per minute when using server key"
    )
    rate_limit_per_hour: int = Field(
        description="Rate limit per hour when using server key"
    )


# Endpoints
@app.get("/api/status", response_model=ApiStatusResponse)
async def api_status() -> ApiStatusResponse:
    """Check if server has an API key configured.

    This allows the frontend to know if users need to provide their own key.
    """
    settings = get_settings()
    rate_limiter = get_rate_limiter()

    return ApiStatusResponse(
        has_server_key=bool(settings.anthropic_api_key),
        rate_limit_per_minute=rate_limiter.requests_per_minute,
        rate_limit_per_hour=rate_limiter.requests_per_hour,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and catalogue status."""
    global _catalogue

    if _catalogue is None:
        try:
            _catalogue = get_catalogue()
        except Exception as e:
            return HealthResponse(status="degraded", catalogue_version=None)

    return HealthResponse(
        status="healthy",
        catalogue_version=_catalogue.version,
        pulsar_count=len(_catalogue._df) if _catalogue._df is not None else None,
    )


@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest) -> QueryResponse:
    """Execute a validated DSL query against the catalogue."""
    global _catalogue, _validator

    try:
        result = query_catalogue(request.query_dsl, catalogue=_catalogue)

        response = QueryResponse(
            success=result.success,
            error=result.error,
        )

        if result.success and result.data is not None:
            response.data = result.data.to_dict(orient="records")
            response.result_count = len(result.data)

            if result.provenance:
                response.provenance = result.provenance.to_dict()

            # Add validation warnings
            if _validator and result.provenance:
                query_dsl = QueryDSL.model_validate(request.query_dsl)
                validation = _validator.validate(
                    result.data, result.provenance, query_dsl
                )
                response.warnings = [w.to_dict() for w in validation.warnings]

        return response

    except Exception as e:
        logger.exception("Query execution failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/pulsar")
async def get_pulsar(request: PulsarInfoRequest) -> dict[str, Any]:
    """Get information about a specific pulsar."""
    global _catalogue

    try:
        result = get_pulsar_info(
            request.pulsar_name,
            fields=request.fields,
            catalogue=_catalogue,
        )

        return result.to_dict()

    except Exception as e:
        logger.exception("Pulsar lookup failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/derived")
async def compute_derived(request: DerivedParameterRequest) -> dict[str, Any]:
    """Compute a derived parameter for query results."""
    global _catalogue

    try:
        # First execute the query
        query_result = query_catalogue(request.query_dsl, catalogue=_catalogue)

        if not query_result.success or query_result.data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=query_result.error or "Query failed",
            )

        # Compute the derived parameter
        result = compute_derived_parameter(
            query_result.data,
            request.parameter,
            use_atnf_native=request.use_atnf_native,
        )

        return {
            "parameter": result.parameter,
            "source": result.source,
            "formula": result.formula,
            "assumptions": result.assumptions,
            "values": result.values.tolist(),
            "pulsar_count": len(result.values),
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Derived parameter computation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/analysis/statistics")
async def run_statistical_analysis(request: AnalysisRequest) -> dict[str, Any]:
    """Run statistical analysis on query results."""
    global _catalogue

    try:
        # Execute query
        query_result = query_catalogue(request.query_dsl, catalogue=_catalogue)

        if not query_result.success or query_result.data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=query_result.error or "Query failed",
            )

        # Run analysis
        result = statistical_analysis(query_result.data, request.parameters)

        return result.to_dict()

    except Exception as e:
        logger.exception("Statistical analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/analysis/correlation")
async def run_correlation_analysis(request: CorrelationRequest) -> dict[str, Any]:
    """Run correlation analysis on query results."""
    global _catalogue

    try:
        # Execute query
        query_result = query_catalogue(request.query_dsl, catalogue=_catalogue)

        if not query_result.success or query_result.data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=query_result.error or "Query failed",
            )

        # Run analysis
        result = correlation_analysis(
            query_result.data,
            request.param_x,
            request.param_y,
            use_log=request.use_log,
        )

        return result.to_dict()

    except Exception as e:
        logger.exception("Correlation analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/plot", response_model=PlotResponse)
async def generate_plot(request: PlotRequest) -> PlotResponse:
    """Generate a visualization for query results."""
    global _catalogue

    try:
        # Execute query
        query_result = query_catalogue(request.query_dsl, catalogue=_catalogue)

        if not query_result.success or query_result.data is None:
            return PlotResponse(
                success=False,
                error=query_result.error or "Query failed",
            )

        df = query_result.data
        options = request.options

        # Generate appropriate plot
        if request.plot_type == "pp_diagram":
            result = create_pp_diagram(
                df,
                highlight_groups=options.get("highlight_groups"),
                show_lines=options.get("show_lines", True),
            )
        elif request.plot_type == "histogram":
            result = create_histogram(
                df,
                parameter=options.get("parameter", "P0"),
                log_scale=options.get("log_scale", False),
                nbins=options.get("nbins", 50),
            )
        elif request.plot_type == "scatter":
            result = create_scatter_plot(
                df,
                x_param=options.get("x_param", "P0"),
                y_param=options.get("y_param", "P1"),
                color_by=options.get("color_by"),
                log_x=options.get("log_x", False),
                log_y=options.get("log_y", False),
                show_regression=options.get("show_regression", False),
            )
        elif request.plot_type == "sky":
            result = create_sky_plot(
                df,
                color_by=options.get("color_by"),
                projection=options.get("projection", "mollweide"),
            )
        elif request.plot_type == "comparison":
            result = create_comparison_plot(
                df,
                parameter=options.get("parameter", "P0"),
                group_column=options.get("group_column", "TYPE"),
                plot_type=options.get("comparison_type", "box"),
            )
        else:
            return PlotResponse(
                success=False,
                error=f"Unknown plot type: {request.plot_type}",
            )

        return PlotResponse(
            success=True,
            plot_json=result.figure.to_json() if result.figure else None,
            plot_html=result.figure.to_html(include_plotlyjs="cdn") if result.figure else None,
        )

    except ValueError as e:
        return PlotResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("Plot generation failed")
        return PlotResponse(success=False, error=str(e))


@app.post("/export/code", response_model=CodeExportResponse)
async def export_code(request: CodeExportRequest) -> CodeExportResponse:
    """Export reproducible Python code for a query."""
    try:
        result = generate_query_plan(request.query_dsl)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to generate code"),
            )

        code = result["python_code"]

        # Add analysis code if requested
        if request.include_analysis:
            code += """

# Statistical analysis
import numpy as np

for col in df.select_dtypes(include=[np.number]).columns:
    print(f"\\n{col}:")
    print(f"  Mean: {df[col].mean():.4g}")
    print(f"  Median: {df[col].median():.4g}")
    print(f"  Std: {df[col].std():.4g}")
    print(f"  Missing: {df[col].isna().sum()}")
"""

        return CodeExportResponse(
            python_code=code,
            description=result.get("plan", "Query execution code"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Code export failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/catalogue/info")
async def catalogue_info() -> dict[str, Any]:
    """Get catalogue metadata and available parameters."""
    global _catalogue

    if _catalogue is None:
        try:
            _catalogue = get_catalogue()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Catalogue not available: {e}",
            )

    from atnf_chat.core.schema import SchemaGroundingPack

    schema = SchemaGroundingPack()

    return {
        "catalogue_version": _catalogue.version,
        "total_pulsars": len(_catalogue._df) if _catalogue._df is not None else 0,
        "available_parameters": list(schema.PARAMETERS.keys()),
        "parameter_count": len(schema.PARAMETERS),
        "categories": {
            cat.value: [
                p.code for p in schema.PARAMETERS.values()
                if p.category.value == cat.value
            ]
            for cat in set(p.category for p in schema.PARAMETERS.values())
        },
    }


@app.get("/parameters/{param_code}")
async def get_parameter_info(param_code: str) -> dict[str, Any]:
    """Get information about a specific parameter."""
    from atnf_chat.core.schema import SchemaGroundingPack

    schema = SchemaGroundingPack()

    param = schema.PARAMETERS.get(param_code.upper())
    if param is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown parameter: {param_code}",
        )

    return {
        "code": param.code,
        "description": param.description,
        "unit": param.unit,
        "type": param.param_type.value,
        "category": param.category.value,
        "typical_range": param.typical_range,
        "aliases": schema.ALIASES.get(param.code, []),
    }
