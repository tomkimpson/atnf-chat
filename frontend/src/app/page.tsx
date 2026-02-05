"use client";

import { useState, useCallback } from "react";
import { ApiKeyModal, Chat, Header } from "../components";
import { useApiKey } from "../lib/useApiKey";

export default function Home() {
  const { apiKey, setApiKey, maskedKey, isLoaded, tier, setTier } = useApiKey();
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [chatKey, setChatKey] = useState(0);

  // Show modal on first visit (no tier preference stored yet)
  const shouldPromptForKey =
    isLoaded &&
    typeof window !== "undefined" &&
    !localStorage.getItem("atnf-chat-tier");

  // Reset chat by changing key to force remount
  const handleLogoClick = useCallback(() => {
    setChatKey((prev) => prev + 1);
  }, []);

  return (
    <div className="flex h-screen flex-col bg-gray-50">
      <Header
        tier={tier}
        maskedKey={maskedKey}
        onApiKeyClick={() => setShowApiKeyModal(true)}
        onLogoClick={handleLogoClick}
      />
      <main className="flex-1 overflow-hidden">
        <Chat
          key={chatKey}
          apiKey={apiKey}
          onApiKeyNeeded={() => setShowApiKeyModal(true)}
        />
      </main>

      <ApiKeyModal
        isOpen={showApiKeyModal || shouldPromptForKey}
        onClose={() => setShowApiKeyModal(false)}
        onSave={setApiKey}
        onTierSelect={setTier}
        currentKey={apiKey}
        currentTier={tier}
      />
    </div>
  );
}
