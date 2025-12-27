/**
 * API client for ATNF-Chat backend
 */

import type { ChatRequest, StreamEvent } from "@/types/chat";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * API status response
 */
export interface ApiStatus {
  hasServerKey: boolean;
  rateLimitPerMinute: number;
  rateLimitPerHour: number;
}

/**
 * Check API status including whether server has an API key configured
 */
export async function getApiStatus(): Promise<ApiStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/status`);
    if (!response.ok) {
      // If endpoint doesn't exist (old backend), assume no server key
      return { hasServerKey: false, rateLimitPerMinute: 0, rateLimitPerHour: 0 };
    }
    const data = await response.json();
    return {
      hasServerKey: data.has_server_key,
      rateLimitPerMinute: data.rate_limit_per_minute,
      rateLimitPerHour: data.rate_limit_per_hour,
    };
  } catch {
    // On network error, assume no server key
    return { hasServerKey: false, rateLimitPerMinute: 0, rateLimitPerHour: 0 };
  }
}

/**
 * Send a chat message and receive a streaming response
 */
export async function* streamChat(
  messages: { role: string; content: string }[],
  apiKey?: string
): AsyncGenerator<StreamEvent> {
  const request: ChatRequest = {
    messages,
    stream: true,
  };

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  const response = await fetch(`${API_BASE_URL}/chat/`, {
    method: "POST",
    headers,
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    // Try to get error details from response
    let errorMessage = `Request failed: ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Use default error message
    }

    yield {
      type: "error",
      error: errorMessage,
    };
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    yield { type: "error", error: "No response body" };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data as StreamEvent;
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

/**
 * Send a chat message and receive a non-streaming response
 */
export async function sendChat(
  messages: { role: string; content: string }[]
): Promise<{
  response: string;
  toolCalls: { name: string; input?: Record<string, unknown> }[];
}> {
  const request: ChatRequest = {
    messages,
    stream: false,
  };

  const response = await fetch(`${API_BASE_URL}/chat/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return {
    response: data.response,
    toolCalls: data.tool_calls || [],
  };
}

/**
 * Get catalogue info
 */
export async function getCatalogueInfo(): Promise<{
  version: string;
  total_pulsars: number;
}> {
  const response = await fetch(`${API_BASE_URL}/catalogue/info`);
  if (!response.ok) {
    throw new Error("Failed to fetch catalogue info");
  }
  const data = await response.json();
  return {
    version: data.catalogue_version,
    total_pulsars: data.total_pulsars,
  };
}

/**
 * Get information about a specific pulsar
 */
export async function getPulsarInfo(name: string): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE_URL}/pulsar`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ pulsar_name: name }),
  });
  if (!response.ok) {
    throw new Error(`Pulsar not found: ${name}`);
  }
  return response.json();
}

/**
 * Execute a query and get results
 */
export async function executeQuery(queryDsl: Record<string, unknown>): Promise<{
  success: boolean;
  data?: Record<string, unknown>[];
  error?: string;
  provenance?: Record<string, unknown>;
}> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query_dsl: queryDsl }),
  });

  return response.json();
}

/**
 * Export query as Python code
 */
export async function exportCode(queryDsl: Record<string, unknown>): Promise<{
  success: boolean;
  python_code?: string;
  error?: string;
}> {
  const response = await fetch(`${API_BASE_URL}/export/code`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query_dsl: queryDsl }),
  });

  return response.json();
}
