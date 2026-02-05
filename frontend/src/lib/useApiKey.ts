"use client";

import { useState, useEffect, useCallback } from "react";

const STORAGE_KEY = "atnf-chat-api-key";
const TIER_KEY = "atnf-chat-tier";

export type ApiTier = "free-shared" | "openrouter" | "anthropic";

function detectTier(key: string): ApiTier {
  if (key.startsWith("sk-ant-")) return "anthropic";
  if (key.startsWith("sk-or-")) return "openrouter";
  return "free-shared";
}

export function useApiKey() {
  const [apiKey, setApiKeyState] = useState<string>("");
  const [tier, setTierState] = useState<ApiTier>("free-shared");
  const [isLoaded, setIsLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem(STORAGE_KEY);
      const storedTier = localStorage.getItem(TIER_KEY) as ApiTier | null;
      if (stored) {
        setApiKeyState(stored);
      }
      if (storedTier) {
        setTierState(storedTier);
      }
      setIsLoaded(true);
    }
  }, []);

  // Save to localStorage
  const setApiKey = useCallback((key: string) => {
    const newTier = detectTier(key);
    setApiKeyState(key);
    setTierState(newTier);
    if (typeof window !== "undefined") {
      if (key) {
        localStorage.setItem(STORAGE_KEY, key);
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
      localStorage.setItem(TIER_KEY, newTier);
    }
  }, []);

  // Set tier without a key (for free-shared)
  const setTier = useCallback((newTier: ApiTier) => {
    setTierState(newTier);
    if (typeof window !== "undefined") {
      localStorage.setItem(TIER_KEY, newTier);
    }
  }, []);

  // Clear the key
  const clearApiKey = useCallback(() => {
    setApiKeyState("");
    setTierState("free-shared");
    if (typeof window !== "undefined") {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.setItem(TIER_KEY, "free-shared");
    }
  }, []);

  // Check if key is set
  const hasApiKey = Boolean(apiKey);

  // Get masked version for display
  const maskedKey = apiKey
    ? `${apiKey.slice(0, 7)}...${apiKey.slice(-4)}`
    : "";

  return {
    apiKey,
    setApiKey,
    clearApiKey,
    hasApiKey,
    maskedKey,
    isLoaded,
    tier,
    setTier,
  };
}
