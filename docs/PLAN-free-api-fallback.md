# Plan: Add Free Default LLM API Option via OpenRouter

## Summary

Add OpenRouter as a free default LLM provider, allowing users to use the app immediately without providing an API key. Users can optionally provide their own Anthropic key for better quality/limits.

## Current State

- Users must provide their own Anthropic API key (via frontend modal)
- Key stored in browser localStorage, sent via `X-API-Key` header
- Server can have fallback key in `.env` (`ANTHROPIC_API_KEY`)
- Uses `claude-sonnet-4-20250514` model (hardcoded)
- Tools defined in Anthropic format (`input_schema` key)

## Architecture

### Provider Priority

1. **User's Anthropic key** (via X-API-Key header) → Use Anthropic client
2. **Server's Anthropic key** (via env) → Use Anthropic client
3. **OpenRouter free tier** (embedded or configured) → Use OpenAI-compatible client

### Key Technical Challenges

**1. Tool Format Conversion**

Anthropic tools use `input_schema`, OpenAI/OpenRouter use `parameters`:

```python
# Anthropic format (current)
{"name": "query_catalogue", "description": "...", "input_schema": {...}}

# OpenAI/OpenRouter format (needed)
{"type": "function", "function": {"name": "query_catalogue", "description": "...", "parameters": {...}}}
```

**2. Response Format Differences**

- Anthropic: `response.content[0].text`, tool calls via `content` blocks with `type="tool_use"`
- OpenAI: `response.choices[0].message.content`, tool calls via `message.tool_calls`

**3. Streaming Format**

- Anthropic: Custom event types (`content_block_delta`)
- OpenAI: SSE with `choices[0].delta`

## Implementation Steps

### 1. Add OpenAI SDK dependency

File: `pyproject.toml`
```
openai>=1.0.0
```

### 2. Update configuration

File: `atnf_chat/config.py`
```python
# New fields
openrouter_api_key: str = Field(default="", description="OpenRouter API key for free tier")
default_free_model: str = Field(default="google/gemini-2.5-flash:free", description="Default free model on OpenRouter")
```

### 3. Create provider abstraction layer

New file: `atnf_chat/llm/providers.py`

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class LLMProvider(ABC):
    @abstractmethod
    async def stream_chat(self, messages, tools, system_prompt) -> AsyncGenerator[dict, None]:
        """Stream chat response with tool support."""
        pass

class AnthropicProvider(LLMProvider):
    """Current Anthropic implementation."""
    pass

class OpenRouterProvider(LLMProvider):
    """OpenRouter/OpenAI-compatible implementation."""
    pass
```

### 4. Tool format converter

New file: `atnf_chat/llm/tool_converter.py`

```python
def anthropic_to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
        }
        for tool in anthropic_tools
    ]
```

### 5. Update chat endpoint

File: `atnf_chat/api/chat.py`

```python
def _get_provider(api_key: str | None, settings: Settings) -> LLMProvider:
    """Get appropriate LLM provider based on available keys."""
    if api_key:
        return AnthropicProvider(api_key)
    if settings.anthropic_api_key:
        return AnthropicProvider(settings.anthropic_api_key)
    if settings.openrouter_api_key:
        return OpenRouterProvider(settings.openrouter_api_key, settings.default_free_model)
    raise HTTPException(status_code=400, detail="No LLM provider configured")
```

### 6. Update frontend

File: `frontend/src/components/ApiKeyModal.tsx`
- Make API key optional
- Add messaging: "API key is optional. Without one, you'll use our free tier (rate limited)."
- Show indicator of which mode is active

File: `frontend/src/lib/api.ts`
- Handle optional key state
- Don't require key before sending requests

### 7. Rate limit handling

Add graceful handling of 429 errors from OpenRouter:
- Display user-friendly message about rate limits
- Suggest providing own API key for higher limits

## Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `openai` dependency |
| `atnf_chat/config.py` | Add OpenRouter settings |
| `atnf_chat/llm/providers.py` | **New** - Provider abstraction |
| `atnf_chat/llm/tool_converter.py` | **New** - Tool format conversion |
| `atnf_chat/api/chat.py` | Use provider abstraction, handle both formats |
| `frontend/src/components/ApiKeyModal.tsx` | Make key optional, update UX |
| `frontend/src/lib/api.ts` | Handle optional key |
| `.env.example` | Document new settings |
| `README.md` | Update documentation |

## Free Models to Support

Recommended OpenRouter free models:
- `google/gemini-2.5-flash:free` - Good balance of quality/availability
- `meta-llama/llama-3.3-70b-instruct:free` - Strong open model
- `deepseek/deepseek-r1t2-chimera:free` - Reasoning capabilities

## Free Provider Comparison

| Provider | Free Tier Limits | Quality | API Compatibility |
|----------|------------------|---------|-------------------|
| **OpenRouter** | ~50 req/day | Good (Gemini, Llama, DeepSeek) | OpenAI-compatible |
| **Groq** | 14,400 req/day | Good (Llama 3.3 70B) | OpenAI-compatible |
| **Google Gemini** | 50-1000 RPD depending on model | Good | Google SDK (different) |
| **Together AI** | $25 free credits | Good | OpenAI-compatible |

OpenRouter was chosen because it provides a unified gateway to multiple free models, making it easy to switch models without code changes if one becomes unavailable.

## Verification Plan

1. **Without any API key**: App should use OpenRouter free tier, tool calling should work
2. **With user Anthropic key**: Should use Anthropic as before
3. **Rate limit test**: Hit OpenRouter limits, verify graceful error handling
4. **Tool calling both providers**: Query catalogue, get pulsar info work on both
5. **Streaming**: Verify streaming works with both providers
6. **Frontend**: Modal shows key as optional, indicates active mode

## Design Decisions

- **Embed shared key + user can provide own**: Best of both worlds - zero-friction start, but users can upgrade
- **OpenRouter as provider**: Unified gateway to 30+ free models, easy to switch if one degrades
- **Full feature parity**: Tool calling works on both providers, requires format conversion layer

## Sources

- [Free LLM API Providers 2026](https://www.analyticsvidhya.com/blog/2026/01/top-free-llm-apis/)
- [OpenRouter Free Models](https://openrouter.ai/collections/free-models)
- [OpenRouter Tool Calling Docs](https://openrouter.ai/docs/guides/features/tool-calling)
- [Groq Free Tier Info](https://community.groq.com/t/is-there-a-free-tier-and-what-are-its-limits/790)
- [Gemini Rate Limits 2026](https://www.aifreeapi.com/en/posts/gemini-api-rate-limits-per-tier)
