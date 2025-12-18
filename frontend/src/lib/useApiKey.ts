"use client";

import { useState, useEffect, useCallback } from "react";

const STORAGE_KEY = "atnf-chat-api-key";

export function useApiKey() {
  const [apiKey, setApiKeyState] = useState<string>("");
  const [isLoaded, setIsLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setApiKeyState(stored);
      }
      setIsLoaded(true);
    }
  }, []);

  // Save to localStorage
  const setApiKey = useCallback((key: string) => {
    setApiKeyState(key);
    if (typeof window !== "undefined") {
      if (key) {
        localStorage.setItem(STORAGE_KEY, key);
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  // Clear the key
  const clearApiKey = useCallback(() => {
    setApiKey("");
  }, [setApiKey]);

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
  };
}
