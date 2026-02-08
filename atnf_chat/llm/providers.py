"""LLM provider implementations.

Provides a unified interface for Anthropic (direct) and OpenRouter (free tier)
so the chat endpoint doesn't need to know which backend is in use.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import anthropic
import openai

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from atnf_chat.config import Settings

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response, yielding SSE-formatted strings."""
        ...  # pragma: no cover

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Non-streaming chat. Returns (text_response, tool_calls_list)."""
        ...  # pragma: no cover


class AnthropicProvider(LLMProvider):
    """Provider using the Anthropic API directly."""

    def __init__(
        self,
        api_key: str,
        model: str,
        tool_executor: Callable[[str, dict[str, Any]], str],
    ) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.tool_executor = tool_executor

    @staticmethod
    def _friendly_anthropic_error(exc: anthropic.APIError) -> str:
        """Return a user-facing error message for common Anthropic errors."""
        status = getattr(exc, "status_code", None)
        if status == 401:
            return (
                "Your Anthropic API key is invalid or expired. "
                "Please check your key at console.anthropic.com/settings/keys."
            )
        if status == 429:
            return (
                "You've hit the Anthropic rate limit. "
                "Please wait a moment and try again."
            )
        if status == 529:
            return (
                "The Anthropic API is temporarily overloaded. "
                "Please try again in a few moments."
            )
        return f"API error: {exc.message}"

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        current_messages = list(messages)

        try:
            for round_num in range(MAX_TOOL_ROUNDS):
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=current_messages,
                    tools=tools,
                ) as stream:
                    async for event in stream:
                        if (
                            hasattr(event, "type")
                            and event.type == "content_block_delta"
                            and hasattr(event.delta, "text")
                        ):
                            yield f"data: {json.dumps({'type': 'text', 'content': event.delta.text})}\n\n"

                    final_message = await stream.get_final_message()

                tool_calls = [
                    block
                    for block in final_message.content
                    if hasattr(block, "type") and block.type == "tool_use"
                ]

                if not tool_calls:
                    break

                tool_results = []
                for tool_call in tool_calls:
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_call.name})}\n\n"
                    result = self.tool_executor(tool_call.name, tool_call.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": result,
                        }
                    )

                current_messages = [
                    *current_messages,
                    {"role": "assistant", "content": final_message.content},
                    {"role": "user", "content": tool_results},
                ]

                logger.debug(f"Tool round {round_num + 1} complete, continuing...")
            else:
                logger.warning(f"Hit max tool rounds ({MAX_TOOL_ROUNDS}), stopping")
                yield f"data: {json.dumps({'type': 'warning', 'message': 'Maximum tool call rounds reached'})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except anthropic.APIError as e:
            logger.exception("Anthropic API error during streaming")
            error_msg = self._friendly_anthropic_error(e)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
        except Exception as e:
            logger.exception("Unexpected error during streaming")
            yield f"data: {json.dumps({'type': 'error', 'error': f'An error occurred: {e!s}'})}\n\n"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict[str, Any]]]:
        current_messages = list(messages)
        tool_calls_made: list[dict[str, Any]] = []

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=current_messages,
            tools=tools,
        )

        while response.stop_reason == "tool_use":
            tool_calls = [
                block
                for block in response.content
                if hasattr(block, "type") and block.type == "tool_use"
            ]

            tool_results = []
            for tool_call in tool_calls:
                tool_calls_made.append(
                    {"name": tool_call.name, "input": tool_call.input}
                )
                result = self.tool_executor(tool_call.name, tool_call.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result,
                    }
                )

            current_messages.append({"role": "assistant", "content": response.content})
            current_messages.append({"role": "user", "content": tool_results})

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=current_messages,
                tools=tools,
            )

        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        return final_text, tool_calls_made


class OpenRouterProvider(LLMProvider):
    """Provider using OpenRouter's OpenAI-compatible API (free tier)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        tool_executor: Callable[[str, dict[str, Any]], str],
        *,
        is_shared_key: bool = False,
    ) -> None:
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://atnf-chat.vercel.app",
                "X-Title": "ATNF-Chat",
            },
        )
        self.model = model
        self.tool_executor = tool_executor
        self.is_shared_key = is_shared_key

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format tools to OpenAI-format tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    def _build_messages(
        self, messages: list[dict[str, Any]], system_prompt: str
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages with system prompt."""
        return [{"role": "system", "content": system_prompt}, *messages]

    @staticmethod
    def _accumulate_tool_call_delta(
        acc: dict[int, dict[str, Any]], tc: Any
    ) -> None:
        """Accumulate a streamed tool call delta into the accumulator."""
        idx = tc.index
        if idx not in acc:
            acc[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
        if tc.id:
            acc[idx]["id"] = tc.id
        if tc.function and tc.function.name:
            acc[idx]["name"] = tc.function.name
        if tc.function and tc.function.arguments:
            acc[idx]["arguments"] += tc.function.arguments

    def _rate_limit_hint(self) -> str:
        """Return a user-facing hint for rate-limit errors."""
        if self.is_shared_key:
            return (
                " — This is a shared key with limited capacity."
                " Get your own free OpenRouter key at openrouter.ai/settings/keys"
                " or provide an Anthropic API key for the best experience."
            )
        return (
            " — You've hit your OpenRouter rate limit."
            " Wait a moment and try again, or provide an Anthropic API key"
            " for higher limits."
        )

    def _friendly_error(self, exc: openai.APIError) -> str:
        """Return a user-facing error message for common OpenRouter errors."""
        status = getattr(exc, "status_code", None)

        if status == 401:
            if self.is_shared_key:
                return (
                    "The free tier is temporarily unavailable due to an "
                    "authentication issue. Please try again later, or "
                    "provide your own API key for uninterrupted access."
                )
            return (
                "Your OpenRouter API key appears to be invalid or expired. "
                "Please check your key at openrouter.ai/settings/keys "
                "or provide an Anthropic API key instead."
            )

        if status == 402:
            if self.is_shared_key:
                return (
                    "The free tier has run out of credits and is temporarily "
                    "unavailable. To continue chatting, you can provide your "
                    "own OpenRouter API key (free at openrouter.ai/settings/keys) "
                    "or an Anthropic API key."
                )
            return (
                "Your OpenRouter account has insufficient credits for this "
                "request. Please add credits at openrouter.ai/settings/credits "
                "or provide an Anthropic API key instead."
            )

        if status == 429:
            return self._rate_limit_hint().lstrip(" —")

        return f"API error: {exc}"

    @staticmethod
    def _parse_tool_args(arguments: str) -> dict[str, Any]:
        """Parse JSON tool call arguments, returning empty dict on failure."""
        if not arguments:
            return {}
        try:
            return json.loads(arguments)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _build_assistant_tool_msg(
        content: str, tool_calls_acc: dict[int, dict[str, Any]]
    ) -> dict[str, Any]:
        """Build an OpenAI-format assistant message with tool calls."""
        return {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls_acc.values()
            ],
        }

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        openai_tools = self._convert_tools(tools)
        current_messages = self._build_messages(messages, system_prompt)

        try:
            for round_num in range(MAX_TOOL_ROUNDS):
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=current_messages,
                    tools=openai_tools,
                    max_tokens=max_tokens,
                    stream=True,
                )

                full_content = ""
                # Accumulate tool calls across chunks
                tool_calls_acc: dict[int, dict[str, Any]] = {}

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    if delta.content:
                        full_content += delta.content
                        yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            self._accumulate_tool_call_delta(tool_calls_acc, tc)

                if not tool_calls_acc:
                    break

                current_messages.append(
                    self._build_assistant_tool_msg(full_content, tool_calls_acc)
                )

                # Execute tools and add results
                for tc in tool_calls_acc.values():
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name']})}\n\n"
                    args = self._parse_tool_args(tc["arguments"])
                    result = self.tool_executor(tc["name"], args)
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )

                logger.debug(f"Tool round {round_num + 1} complete, continuing...")
            else:
                logger.warning(f"Hit max tool rounds ({MAX_TOOL_ROUNDS}), stopping")
                yield f"data: {json.dumps({'type': 'warning', 'message': 'Maximum tool call rounds reached'})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except openai.APIError as e:
            logger.exception("OpenRouter API error during streaming")
            yield f"data: {json.dumps({'type': 'error', 'error': self._friendly_error(e)})}\n\n"
        except Exception as e:
            logger.exception("Unexpected error during streaming")
            yield f"data: {json.dumps({'type': 'error', 'error': f'An error occurred: {e!s}'})}\n\n"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict[str, Any]]]:
        openai_tools = self._convert_tools(tools)
        current_messages = self._build_messages(messages, system_prompt)
        tool_calls_made: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                tools=openai_tools,
                max_tokens=max_tokens,
            )

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                return msg.content or "", tool_calls_made

            # Build assistant message for history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            current_messages.append(assistant_msg)

            for tc in msg.tool_calls:
                args = self._parse_tool_args(tc.function.arguments)
                tool_calls_made.append({"name": tc.function.name, "input": args})
                result = self.tool_executor(tc.function.name, args)
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return "", tool_calls_made


def get_provider(
    api_key: str | None,
    settings: Settings,
    tool_executor: Callable[[str, dict[str, Any]], str],
) -> LLMProvider:
    """Create the appropriate LLM provider based on available credentials.

    Priority: user API key (detected by prefix) → server Anthropic key
    → server OpenRouter key (shared free tier).
    """
    if api_key:
        if api_key.startswith("sk-ant-"):
            return AnthropicProvider(api_key, settings.anthropic_model, tool_executor)
        if api_key.startswith("sk-or-"):
            return OpenRouterProvider(
                api_key, settings.openrouter_model, tool_executor
            )
        raise ValueError(
            "Unrecognized API key format. "
            "Anthropic keys start with 'sk-ant-', OpenRouter keys start with 'sk-or-'."
        )
    if settings.anthropic_api_key:
        return AnthropicProvider(
            settings.anthropic_api_key, settings.anthropic_model, tool_executor
        )
    if settings.openrouter_api_key:
        return OpenRouterProvider(
            settings.openrouter_api_key,
            settings.openrouter_model,
            tool_executor,
            is_shared_key=True,
        )
    raise ValueError("No LLM provider configured")
