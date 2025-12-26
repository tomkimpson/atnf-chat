# Vercel AI Gateway Integration Plan

## Current Architecture Analysis

The current implementation uses a **three-tier architecture**:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Next.js Frontend   │────▶│  FastAPI Backend    │────▶│  Anthropic API      │
│  (React/TypeScript) │     │  (Python)           │     │  (Claude)           │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
        │                           │
        │ API key in localStorage   │ Uses X-API-Key header
        │ Sent via HTTP header      │ or server env var
        └───────────────────────────┘
```

**Key files:**
- `frontend/src/lib/useApiKey.ts` - localStorage-based key storage
- `frontend/src/lib/api.ts:25-26` - Sends key as `X-API-Key` header
- `atnf_chat/api/chat.py:287-288` - Backend prioritizes client key, falls back to server env var

---

## Evaluation: Vercel AI Gateway vs User-Provided Keys

### Option A: Vercel AI Gateway (Server-Side Key)

**How it works:**
- Store `ANTHROPIC_API_KEY` as a Vercel environment variable
- Requests are proxied through Vercel's infrastructure
- Users never see or manage API keys

**Pros:**
- ✅ Zero friction for users - no API key setup required
- ✅ Centralized key management and monitoring
- ✅ Usage analytics and cost tracking through Vercel dashboard
- ✅ No risk of users leaking their own keys
- ✅ Better security - keys never touch browser

**Cons:**
- ⚠️ **Architecture mismatch**: Vercel AI Gateway is designed for Node.js/Edge functions, not Python backends
- ⚠️ You pay for all API usage (no cost distribution to users)
- ⚠️ Need rate limiting to prevent abuse
- ⚠️ Requires migration from Python backend to Next.js API routes for LLM calls

### Option B: User-Provided Keys Only (Current)

**Pros:**
- ✅ No cost to project maintainer
- ✅ Users control their own usage and billing
- ✅ Works with current Python backend architecture

**Cons:**
- ❌ High friction - users must obtain and manage keys
- ❌ Keys stored in browser localStorage (less secure)
- ❌ Keys transmitted over network (even via HTTPS)
- ❌ Barrier to entry for casual users

### Option C: Hybrid Approach (Recommended)

**Server-managed key as primary, user key as fallback:**

```
┌─────────────────────────────────────────────────────────────┐
│                      User Request                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Has server key │──Yes──▶ Use server key (default)
                    │  configured?    │         (seamless experience)
                    └─────────────────┘
                              │ No
                              ▼
                    ┌─────────────────┐
                    │ User provided   │──Yes──▶ Use user's key
                    │ their own key?  │         (power user mode)
                    └─────────────────┘
                              │ No
                              ▼
                    ┌─────────────────┐
                    │ Prompt user to  │
                    │ enter their key │
                    └─────────────────┘
```

**Pros:**
- ✅ Seamless for most users (when server key is configured)
- ✅ Power users can use their own keys (for privacy or higher limits)
- ✅ Works with current Python backend (no architecture change)
- ✅ Graceful degradation if server key quota is exhausted

---

## Recommended Implementation: Hybrid Approach

Given that this project uses a **Python/FastAPI backend**, the Vercel AI Gateway isn't directly applicable (it's designed for Node.js). Instead, we can achieve the same goal by:

1. **Configure server-side API key** in Vercel/Railway environment
2. **Check key availability on startup** and inform frontend
3. **Adjust UI** to make API key entry optional
4. **Add rate limiting** to prevent abuse

### Implementation Steps

#### Phase 1: Backend Changes

**Step 1.1**: Add endpoint to check if server has API key configured
- New endpoint: `GET /config/status`
- Returns: `{ "server_api_key_available": true/false, "rate_limit_info": {...} }`

**Step 1.2**: Update chat endpoint priority logic
- Current: client key OR server key
- New: server key (if available) OR client key
- This makes server key the default for seamless UX

**Step 1.3**: Add rate limiting (essential when using shared key)
- IP-based rate limiting
- Configurable via environment variables
- Return 429 with helpful message when limit reached

#### Phase 2: Frontend Changes

**Step 2.1**: Fetch server config on load
- Call `/config/status` to check if server key is available
- Store result in app state

**Step 2.2**: Update UI based on server key availability
```
If server key available:
  - Show "Ready to chat" - no API key prompt
  - Add subtle "Use your own key" option in settings

If no server key:
  - Show current API key modal (existing behavior)
```

**Step 2.3**: Update API calls
- Don't send `X-API-Key` header if using server key
- Only send header when user explicitly provides their own key

**Step 2.4**: Add settings toggle for "Use my own API key"
- Allows power users to override server key
- Useful for higher rate limits or privacy

#### Phase 3: Configuration & Documentation

**Step 3.1**: Update environment variables
```env
# Server-managed API key (optional, enables zero-config for users)
ANTHROPIC_API_KEY=sk-ant-...

# Rate limiting (required when using server key)
RATE_LIMIT_REQUESTS_PER_MINUTE=20
RATE_LIMIT_REQUESTS_PER_DAY=100
```

**Step 3.2**: Update deployment documentation
- Document how to configure server key
- Explain rate limiting configuration
- Document cost implications

---

## Files to Modify

| File | Changes |
|------|---------|
| `atnf_chat/api/config_routes.py` | New file: config status endpoint |
| `atnf_chat/api/chat.py` | Update key priority, add rate limiting |
| `atnf_chat/config.py` | Add rate limit settings |
| `frontend/src/lib/api.ts` | Add config fetch, conditional header |
| `frontend/src/lib/useApiKey.ts` | Add server key awareness |
| `frontend/src/app/page.tsx` | Conditional API key modal |
| `frontend/src/components/Header.tsx` | Update API key button state |
| `frontend/src/components/ApiKeyModal.tsx` | Add "use your own key" option |
| `.env.example` | Document new variables |

---

## Questions for Validation

1. **Cost tolerance**: Are you willing to pay for API usage when users don't provide their own keys? (Required for seamless UX)

2. **Rate limits**: What default limits make sense?
   - Suggested: 20 requests/minute, 100 requests/day per IP

3. **Priority**: Which approach for key selection?
   - A) Server key preferred (seamless default)
   - B) User key preferred (more control to users)

4. **Fallback behavior**: When server key is unavailable, should we:
   - A) Require user key (current behavior)
   - B) Disable chat with helpful message

---

## Alternative: Pure Vercel AI Gateway

If you want true Vercel AI Gateway integration, this would require:

1. **Move LLM calls to Next.js API routes** (replacing Python for chat)
2. **Keep Python backend** only for catalogue queries and tools
3. **Use Vercel AI SDK** in Next.js for streaming

This is a larger architectural change but provides:
- Native Vercel AI Gateway support
- Edge function performance
- Built-in streaming helpers

**Effort estimate**: Significant refactor of chat functionality

---

## Recommendation

Start with **Hybrid Approach (Option C)** because:
1. Works with existing Python architecture
2. Minimal code changes
3. Achieves the main goal (seamless UX for users)
4. Preserves flexibility for power users
5. Can evaluate Vercel AI Gateway migration later if needed
