"""Integration between benchmarks and the ATNF-Chat LLM client.

This module provides the connection between the benchmark framework
and the actual LLM-powered chat endpoint.

Example:
    >>> from benchmarks.integration import LLMBenchmarkClient
    >>> client = LLMBenchmarkClient()
    >>> response = client.query("Show me millisecond pulsars")
    >>> print(response)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from atnf_chat.config import get_settings
from atnf_chat.core.catalogue import get_catalogue
from atnf_chat.core.dsl import QueryDSL
from atnf_chat.tools import (
    compute_derived_parameter,
    correlation_analysis,
    get_pulsar_info,
    query_catalogue,
    statistical_analysis,
)

logger = logging.getLogger(__name__)


class LLMBenchmarkClient:
    """Client for running benchmarks against the real LLM.

    This can operate in two modes:
    1. Direct mode: Calls tools directly with LLM-generated DSL
    2. API mode: Calls the full chat API endpoint

    Direct mode is faster and doesn't require the API server to be running.
    """

    def __init__(
        self,
        api_url: str | None = None,
        use_api: bool = False,
    ) -> None:
        """Initialize the benchmark client.

        Args:
            api_url: URL of the chat API (for API mode)
            use_api: Use the full API instead of direct tool calls
        """
        self.api_url = api_url or "http://localhost:8000"
        self.use_api = use_api
        self._catalogue = None

        # Initialize Anthropic client for direct mode
        settings = get_settings()
        if settings.anthropic_api_key:
            import anthropic

            self.anthropic_client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key
            )
        else:
            self.anthropic_client = None

    @property
    def catalogue(self):
        """Lazy-load catalogue."""
        if self._catalogue is None:
            self._catalogue = get_catalogue()
        return self._catalogue

    def query(self, user_query: str) -> dict[str, Any]:
        """Execute a benchmark query and get structured response.

        Args:
            user_query: The natural language query

        Returns:
            Dict with answer, tool_calls, query_dsl, provenance
        """
        if self.use_api:
            return self._query_via_api(user_query)
        else:
            return self._query_direct(user_query)

    def _query_via_api(self, user_query: str) -> dict[str, Any]:
        """Query via the HTTP API.

        Args:
            user_query: The natural language query

        Returns:
            Response dict
        """
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.api_url}/api/chat/",
                    json={
                        "messages": [{"role": "user", "content": user_query}],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

                return {
                    "answer": data.get("response", ""),
                    "tool_calls": data.get("tool_calls", []),
                    "query_dsl": self._extract_dsl_from_tool_calls(
                        data.get("tool_calls", [])
                    ),
                    "provenance": None,  # Would need to extract from response
                }

        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            return {
                "answer": "",
                "tool_calls": [],
                "query_dsl": None,
                "provenance": None,
                "error": str(e),
            }

    def _query_direct(self, user_query: str) -> dict[str, Any]:
        """Query directly using Anthropic client and tools.

        Args:
            user_query: The natural language query

        Returns:
            Response dict
        """
        if self.anthropic_client is None:
            return {
                "answer": "",
                "tool_calls": [],
                "query_dsl": None,
                "provenance": None,
                "error": "Anthropic client not configured",
            }

        from atnf_chat.tools import get_tools_for_claude

        tools = get_tools_for_claude()
        system_prompt = self._build_benchmark_system_prompt()

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_query}],
                tools=tools,
            )

            # Extract tool calls
            tool_calls = []
            query_dsl = None
            provenance = None

            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_call = {
                        "name": block.name,
                        "input": block.input,
                    }
                    tool_calls.append(tool_call)

                    # Extract DSL if it's a query
                    if block.name == "query_catalogue":
                        # The tool schema defines DSL fields at the top level,
                        # so the entire input dict IS the query DSL.
                        query_dsl = block.input

                        # Actually execute to get provenance
                        result = self._execute_tool(block.name, block.input)
                        if hasattr(result, "provenance") and result.provenance is not None:
                            provenance = result.provenance.to_dict()

            # Get text response
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer += block.text

            # If there were tool calls, continue conversation
            if tool_calls and response.stop_reason == "tool_use":
                # Execute tools and get final response
                continuation = self._continue_with_tools(
                    response, user_query, tools, system_prompt
                )
                answer = continuation.get("answer", answer)
                if continuation.get("provenance"):
                    provenance = continuation["provenance"]

            return {
                "answer": answer,
                "tool_calls": tool_calls,
                "query_dsl": query_dsl,
                "provenance": provenance,
            }

        except Exception as e:
            logger.exception("Direct query failed")
            return {
                "answer": "",
                "tool_calls": [],
                "query_dsl": None,
                "provenance": None,
                "error": str(e),
            }

    def _continue_with_tools(
        self,
        initial_response,
        user_query: str,
        tools: list[dict],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Continue conversation after tool use.

        Args:
            initial_response: The initial LLM response with tool calls
            user_query: Original user query
            tools: Tool definitions
            system_prompt: System prompt

        Returns:
            Final response dict
        """
        messages = [{"role": "user", "content": user_query}]

        # Add assistant response with tool calls
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute tools and add results
        tool_results = []
        provenance = None

        for block in initial_response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                result = self._execute_tool(block.name, block.input)

                # Get provenance if available
                if result is not None and hasattr(result, "provenance") and result.provenance is not None:
                    provenance = result.provenance.to_dict()

                content = "No result"
                if result is not None:
                    content = result.format_for_display() if hasattr(result, "format_for_display") else str(result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })

        messages.append({"role": "user", "content": tool_results})

        # Get final response
        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )

        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer += block.text

        return {
            "answer": answer,
            "provenance": provenance,
        }

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute a tool with given input.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters

        Returns:
            Tool result
        """
        catalogue = self.catalogue

        if tool_name == "query_catalogue":
            return query_catalogue(
                tool_input.get("query_dsl", tool_input),
                catalogue=catalogue,
            )

        elif tool_name == "get_pulsar_info":
            return get_pulsar_info(
                tool_input.get("name", tool_input.get("pulsar_name", "")),
                fields=tool_input.get("fields"),
                catalogue=catalogue,
            )

        elif tool_name == "compute_derived_parameter":
            # First get data if needed
            query_dsl = tool_input.get("query_dsl")
            if query_dsl:
                query_result = query_catalogue(query_dsl, catalogue=catalogue)
                if query_result.success and query_result.data is not None:
                    return compute_derived_parameter(
                        query_result.data,
                        tool_input.get("parameter", ""),
                        use_atnf_native=tool_input.get("use_atnf_native", True),
                    )
            return None

        elif tool_name == "statistical_analysis":
            query_dsl = tool_input.get("query_dsl")
            if query_dsl:
                query_result = query_catalogue(query_dsl, catalogue=catalogue)
                if query_result.success and query_result.data is not None:
                    return statistical_analysis(
                        query_result.data,
                        tool_input.get("parameters"),
                    )
            return None

        elif tool_name == "correlation_analysis":
            query_dsl = tool_input.get("query_dsl")
            if query_dsl:
                query_result = query_catalogue(query_dsl, catalogue=catalogue)
                if query_result.success and query_result.data is not None:
                    return correlation_analysis(
                        query_result.data,
                        tool_input.get("param_x", ""),
                        tool_input.get("param_y", ""),
                        use_log=tool_input.get("use_log", False),
                    )
            return None

        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return None

    def _extract_dsl_from_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Extract query DSL from tool calls."""
        for call in tool_calls:
            if call.get("name") == "query_catalogue":
                return call.get("input", {}).get("query_dsl")
        return None

    def _build_benchmark_system_prompt(self) -> str:
        """Build system prompt for benchmarking."""
        return """You are ATNF-Chat, an expert assistant for querying the ATNF Pulsar Catalogue.

For benchmark evaluation, please:
1. Use the query_catalogue tool for all data queries
2. Use proper DSL format with validated fields
3. Include relevant provenance information
4. Handle edge cases gracefully

When asked about specific pulsars, use get_pulsar_info.
When asked for statistics, use statistical_analysis.
When asked about correlations, use correlation_analysis.
When asked about derived parameters, use compute_derived_parameter.

Always use official ATNF parameter codes (P0, F0, DM, BSURF, etc.).
"""


def run_live_benchmark(
    test_ids: list[str] | None = None,
    output_file: str | None = None,
) -> None:
    """Run benchmarks against the live LLM.

    Args:
        test_ids: Specific test IDs to run (None = all)
        output_file: Path to save results JSON
    """
    from benchmarks.evaluate import BenchmarkRunner, BENCHMARK_FILE

    # Create client
    client = LLMBenchmarkClient(use_api=False)

    # Create custom runner that uses the real client
    class LiveBenchmarkRunner(BenchmarkRunner):
        def __init__(self, client: LLMBenchmarkClient):
            super().__init__(llm_client=client)
            self.client = client

        def _get_llm_response(self, query: str) -> dict[str, Any]:
            return self.client.query(query)

    runner = LiveBenchmarkRunner(client)

    # Run benchmarks
    print("Running live benchmarks against LLM...")
    print("This may take several minutes and incur API costs.\n")

    results = runner.run_all(skip_llm=False)

    print(results.summary())

    if output_file:
        import json
        from pathlib import Path

        with open(output_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else None
    run_live_benchmark(output_file=output)
