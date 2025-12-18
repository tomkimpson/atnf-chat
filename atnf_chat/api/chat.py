"""Chat endpoint with LLM integration and streaming.

This module provides the conversational interface to the ATNF catalogue,
using Claude for natural language understanding and function calling.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

import anthropic
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from atnf_chat.config import get_settings
from atnf_chat.core.catalogue import get_catalogue
from atnf_chat.core.schema import SchemaGroundingPack
from atnf_chat.tools import (
    compute_derived_parameter,
    correlation_analysis,
    generate_query_plan,
    get_pulsar_info,
    get_tools_for_claude,
    query_catalogue,
    statistical_analysis,
)

if TYPE_CHECKING:
    from anthropic.types import Message, MessageStreamEvent

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

3. **Derived Parameters**: For quantities like BSURF, EDOT, AGE:
   - Prefer ATNF-native values when available (set use_atnf_native=True)
   - Document assumptions when computing (moment of inertia, braking index)

4. **Scientific Rigor**:
   - Report data completeness (mention when fields have high missingness)
   - Note selection effects when filtering
   - Include provenance information for reproducibility

5. **Result Formatting**:
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


async def _stream_chat_response(
    client: anthropic.AsyncAnthropic,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Stream chat response with tool handling.

    Supports multiple rounds of tool calls - the LLM can make sequential
    tool calls (e.g., query + calculate percentage) and all will be processed.
    """
    system_prompt = _build_system_prompt()
    current_messages = list(messages)  # Copy to avoid mutation
    max_tool_rounds = 10  # Safety limit to prevent infinite loops

    try:
        for round_num in range(max_tool_rounds):
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=current_messages,
                tools=tools,
            ) as stream:
                # Stream text as it arrives
                async for event in stream:
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield f"data: {json.dumps({'type': 'text', 'content': event.delta.text})}\n\n"

                # Get the final message to check for tool use
                final_message = await stream.get_final_message()

            # Check if there are tool calls to process
            tool_calls = [
                block for block in final_message.content
                if hasattr(block, "type") and block.type == "tool_use"
            ]

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Process tool calls
            tool_results = []
            for tool_call in tool_calls:
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_call.name})}\n\n"

                result = _execute_tool(tool_call.name, tool_call.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                })

            # Update messages for next round
            current_messages = current_messages + [
                {"role": "assistant", "content": final_message.content},
                {"role": "user", "content": tool_results},
            ]

            logger.debug(f"Tool round {round_num + 1} complete, continuing...")

        else:
            # Hit max rounds - log warning
            logger.warning(f"Hit max tool rounds ({max_tool_rounds}), stopping")
            yield f"data: {json.dumps({'type': 'warning', 'message': 'Maximum tool call rounds reached'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except anthropic.APIError as e:
        logger.exception("Anthropic API error during streaming")
        yield f"data: {json.dumps({'type': 'error', 'error': f'API error: {e.message}'})}\n\n"
    except Exception as e:
        logger.exception("Unexpected error during streaming")
        yield f"data: {json.dumps({'type': 'error', 'error': f'An error occurred: {str(e)}'})}\n\n"


@router.post("/", response_model=None)
async def chat(
    request: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> StreamingResponse | ChatResponse:
    """Process a chat message with optional streaming."""
    settings = get_settings()

    # Use client-provided API key first, fall back to server-configured key
    api_key = x_api_key or settings.anthropic_api_key

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Please provide your Anthropic API key.",
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)
    tools = get_tools_for_claude()

    # Convert messages to API format
    api_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]

    if request.stream:
        return StreamingResponse(
            _stream_chat_response(client, api_messages, tools),
            media_type="text/event-stream",
        )

    # Non-streaming response
    system_prompt = _build_system_prompt()

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=api_messages,
        tools=tools,
    )

    # Handle tool calls
    tool_calls_made = []
    while response.stop_reason == "tool_use":
        tool_calls = [
            block for block in response.content
            if hasattr(block, "type") and block.type == "tool_use"
        ]

        tool_results = []
        for tool_call in tool_calls:
            tool_calls_made.append({
                "name": tool_call.name,
                "input": tool_call.input,
            })

            result = _execute_tool(tool_call.name, tool_call.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result,
            })

        # Continue with tool results
        api_messages.append({"role": "assistant", "content": response.content})
        api_messages.append({"role": "user", "content": tool_results})

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=api_messages,
            tools=tools,
        )

    # Extract final text response
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    return ChatResponse(
        response=final_text,
        tool_calls=tool_calls_made,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )


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
