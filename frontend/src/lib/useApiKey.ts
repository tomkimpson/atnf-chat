"use client";

import { useState, useEffect, useCallback } from "react";
import { getApiStatus, type ApiStatus } from "./api";

const STORAGE_KEY = "atnf-chat-api-key";

export function useApiKey() {
  const [apiKey, setApiKeyState] = useState<string>("");
  const [isLoaded, setIsLoaded] = useState(false);
  const [serverStatus, setServerStatus] = useState<ApiStatus | null>(null);

  // Load from localStorage and check server status on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setApiKeyState(stored);
      }
      setIsLoaded(true);

      // Check if server has an API key configured
      getApiStatus().then(setServerStatus);
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

  // Check if API key is required (no server key and no user key)
  const isApiKeyRequired = serverStatus !== null && !serverStatus.hasServerKey && !hasApiKey;

  // Check if server has a key configured
  const serverHasApiKey = serverStatus?.hasServerKey ?? false;

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
    serverHasApiKey,
    isApiKeyRequired,
    rateLimitPerMinute: serverStatus?.rateLimitPerMinute ?? 0,
    rateLimitPerHour: serverStatus?.rateLimitPerHour ?? 0,
  };
}
