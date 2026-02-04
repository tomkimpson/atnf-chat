"""Chat endpoint with LLM integration and streaming.

This module provides the conversational interface to the ATNF catalogue,
using an LLM provider (Anthropic or OpenRouter) for natural language
understanding and function calling.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from atnf_chat.config import get_settings
from atnf_chat.core.catalogue import get_catalogue
from atnf_chat.core.schema import SchemaGroundingPack
from atnf_chat.llm import get_provider
from atnf_chat.tools import (
    compute_derived_parameter,
    correlation_analysis,
    generate_query_plan,
    get_pulsar_info,
    get_tools_for_claude,
    query_catalogue,
    statistical_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    messages: list[ChatMessage] = Field(..., description="Conversation history")
    stream: bool = Field(default=True, description="Whether to stream response")


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] | None = None


def _build_system_prompt() -> str:
    """Build the system prompt with schema grounding."""
    schema = SchemaGroundingPack()

    # Build parameter reference
    param_lines = []
    for code, param in schema.PARAMETERS.items():
        aliases = schema.get_aliases(code)
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        param_lines.append(
            f"- {code}: {param.description} [{param.unit or 'dimensionless'}]{alias_str}"
        )

    param_reference = "\n".join(param_lines[:30])  # Truncate for token efficiency

    return f"""You are ATNF-Chat, an expert assistant for querying the ATNF Pulsar Catalogue.

You help astronomers and researchers explore pulsar data using natural language queries.
You have access to tools for querying the catalogue, computing derived parameters,
and performing statistical analyses.

## Key Guidelines

1. **Query Translation**: Convert natural language to structured DSL queries.
   Use the query_catalogue tool with proper field names and filters.

2. **Field Names**: Always use official ATNF parameter codes (P0, F0, DM, etc.).
   Common mappings:
   - "period" → P0 (barycentric period in seconds)
   - "frequency" → F0 (spin frequency in Hz)
   - "dispersion measure" or "DM" → DM
   - "distance" → DIST (in kpc)
   - "magnetic field" → BSURF (surface B-field in Gauss)
   - "spin-down" or "period derivative" → P1

3. **Standard Pulsar Classifications**: Use these conventional definitions unless
   the user specifies otherwise:
   - "millisecond pulsar" (MSP) → P0 < 0.03 s (30 ms)
   - "binary pulsar" → PB is not null (has a measured orbital period)
   - "magnetar" → BSURF > 1e14 Gauss
   - "recycled pulsar" → P0 < 0.03 s and P1 < 1e-17
   - "young pulsar" → characteristic age < 100 kyr

4. **Derived Parameters**: For quantities like BSURF, EDOT, AGE:
   - Prefer ATNF-native values when available (set use_atnf_native=True)
   - Document assumptions when computing (moment of inertia, braking index)

5. **Scientific Rigor**:
   - Report data completeness (mention when fields have high missingness)
   - Note selection effects when filtering
   - Include provenance information for reproducibility

6. **Result Formatting**:
   - For small results (≤10 rows), show a table
   - For larger results, summarize and offer to show samples
   - Include units in all numerical results

## Available Parameters (subset)

{param_reference}

## Tool Usage

- query_catalogue: Execute validated queries against the catalogue
- get_pulsar_info: Look up a specific pulsar by name
- compute_derived_parameter: Calculate BSURF, EDOT, AGE, etc.
- statistical_analysis: Get summary statistics
- correlation_analysis: Analyze relationships between parameters
- generate_query_plan: Show reproducible Python code

Always validate field names before querying. If a user asks for an unknown field,
suggest the closest valid parameter.
"""


def _execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool and return the result as a string."""
    catalogue = get_catalogue()

    try:
        if tool_name == "query_catalogue":
            # The tool input IS the query_dsl - properties are at top level
            result = query_catalogue(tool_input, catalogue=catalogue)
            return result.format_for_display()

        elif tool_name == "get_pulsar_info":
            result = get_pulsar_info(
                tool_input.get("name", ""),
                catalogue=catalogue,
            )
            return result.format_for_display()

        elif tool_name == "compute_derived_parameter":
            # First get the data
            query_result = query_catalogue(
                tool_input.get("query_dsl", {}), catalogue=catalogue
            )
            if not query_result.success or query_result.data is None:
                return f"Query failed: {query_result.error}"

            result = compute_derived_parameter(
                query_result.data,
                tool_input.get("parameter", ""),
                use_atnf_native=tool_input.get("use_atnf_native", True),
            )
            return result.format_for_display()

        elif tool_name == "statistical_analysis":
            query_result = query_catalogue(
                tool_input.get("query_dsl", {}), catalogue=catalogue
            )
            if not query_result.success or query_result.data is None:
                return f"Query failed: {query_result.error}"

            result = statistical_analysis(
                query_result.data, tool_input.get("parameters")
            )
            return result.format_for_display()

        elif tool_name == "correlation_analysis":
            query_result = query_catalogue(
                tool_input.get("query_dsl", {}), catalogue=catalogue
            )
            if not query_result.success or query_result.data is None:
                return f"Query failed: {query_result.error}"

            result = correlation_analysis(
                query_result.data,
                tool_input.get("param_x", ""),
                tool_input.get("param_y", ""),
                use_log=tool_input.get("use_log", False),
            )
            return result.format_for_display()

        elif tool_name == "generate_query_plan":
            result = generate_query_plan(tool_input.get("query_dsl", {}))
            if result["success"]:
                return f"Query Plan:\n{result['plan']}\n\nPython Code:\n```python\n{result['python_code']}\n```"
            else:
                return f"Failed to generate plan: {result.get('error', 'Unknown error')}"

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as e:
        logger.exception(f"Tool execution failed: {tool_name}")
        return f"Tool execution failed: {e}"


@router.post("/", response_model=None)
async def chat(
    request: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> StreamingResponse | ChatResponse:
    """Process a chat message with optional streaming."""
    settings = get_settings()
    tools = get_tools_for_claude()

    try:
        provider = get_provider(x_api_key, settings, _execute_tool)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="No LLM provider configured. Set OPENROUTER_API_KEY or provide an Anthropic key.",
        ) from exc

    api_messages: list[dict[str, Any]] = [
        {"role": msg.role, "content": msg.content} for msg in request.messages
    ]

    if request.stream:
        return StreamingResponse(
            provider.stream_chat(api_messages, tools, _build_system_prompt()),
            media_type="text/event-stream",
        )

    text, tool_calls = await provider.chat(
        api_messages, tools, _build_system_prompt()
    )
    return ChatResponse(response=text, tool_calls=tool_calls)


@router.post("/complete")
async def complete_query(query: str) -> dict[str, Any]:
    """Get query suggestions for partial input."""
    schema = SchemaGroundingPack()

    # Simple completion based on parameter matching
    suggestions = []
    query_lower = query.lower()

    for code, param in schema.PARAMETERS.items():
        if query_lower in code.lower() or query_lower in param.description.lower():
            suggestions.append({
                "code": code,
                "description": param.description,
                "unit": param.unit,
            })

    # Also check aliases
    for alias, code in schema._alias_to_code.items():
        if query_lower in alias.lower():
            param = schema.PARAMETERS[code]
            suggestions.append({
                "code": code,
                "description": param.description,
                "unit": param.unit,
                "matched_alias": alias,
            })

    # Deduplicate and limit
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s["code"] not in seen:
            seen.add(s["code"])
            unique_suggestions.append(s)

    return {
        "query": query,
        "suggestions": unique_suggestions[:10],
    }
