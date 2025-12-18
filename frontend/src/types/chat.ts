/**
 * Types for the chat interface
 */

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
}

export interface ToolCall {
  name: string;
  input?: Record<string, unknown>;
  result?: string;
}

export interface ChatRequest {
  messages: { role: string; content: string }[];
  stream: boolean;
}

export interface ChatResponse {
  response: string;
  tool_calls: ToolCall[];
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
}

export interface StreamEvent {
  type: "text" | "tool_call" | "done" | "error";
  content?: string;
  name?: string;
  error?: string;
}

export interface PlotData {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
}

export interface QueryResult {
  success: boolean;
  data?: Record<string, unknown>[];
  error?: string;
  provenance?: {
    catalogue_version: string;
    snapshot_date: string;
    result_count: number;
    null_counts: Record<string, number>;
  };
}
