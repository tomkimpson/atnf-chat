"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { WelcomeScreen } from "./WelcomeScreen";
import { streamChat } from "../lib/api";
import type { Message, ToolCall } from "../types/chat";

function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

interface ChatProps {
  apiKey: string;
  onApiKeyNeeded: () => void;
}

export function Chat({ apiKey, onApiKeyNeeded }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSend = async (content: string) => {
    // Check if API key is set
    if (!apiKey) {
      onApiKeyNeeded();
      return;
    }

    // Add user message
    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Create assistant message placeholder
    const assistantMessage: Message = {
      id: generateId(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
      toolCalls: [],
    };

    setMessages((prev) => [...prev, assistantMessage]);

    try {
      // Build conversation history for API
      const conversationHistory = [...messages, userMessage].map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      // Stream the response
      let fullContent = "";
      const toolCalls: ToolCall[] = [];

      for await (const event of streamChat(conversationHistory, apiKey)) {
        switch (event.type) {
          case "text":
            fullContent += event.content || "";
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessage.id
                  ? { ...msg, content: fullContent }
                  : msg
              )
            );
            break;

          case "tool_call":
            toolCalls.push({ name: event.name || "unknown" });
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessage.id
                  ? { ...msg, toolCalls: [...toolCalls] }
                  : msg
              )
            );
            break;

          case "error":
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessage.id
                  ? {
                      ...msg,
                      content:
                        msg.content +
                        `\n\n**Error:** ${event.error || "Unknown error"}`,
                      isStreaming: false,
                    }
                  : msg
              )
            );
            break;

          case "done":
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessage.id
                  ? { ...msg, isStreaming: false }
                  : msg
              )
            );
            break;
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content:
                  "Sorry, I encountered an error connecting to the server. Please make sure the API server is running on http://localhost:8000",
                isStreaming: false,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (example: string) => {
    handleSend(example);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <WelcomeScreen onExampleClick={handleExampleClick} />
        ) : (
          <div className="mx-auto max-w-4xl">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <ChatInput
        onSend={handleSend}
        disabled={isLoading}
        placeholder={
          isLoading
            ? "Waiting for response..."
            : "Ask about pulsars, e.g., 'Show me millisecond pulsars'"
        }
      />
    </div>
  );
}
