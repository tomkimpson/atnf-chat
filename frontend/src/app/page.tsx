"use client";

import { useState, useCallback } from "react";
import { ApiKeyModal, Chat, Header } from "../components";
import { useApiKey } from "../lib/useApiKey";

export default function Home() {
  const { apiKey, setApiKey, hasApiKey, maskedKey, isLoaded } = useApiKey();
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [chatKey, setChatKey] = useState(0);

  // Show modal automatically if no API key is set (after loading)
  const shouldPromptForKey = isLoaded && !hasApiKey;

  // Reset chat by changing key to force remount
  const handleLogoClick = useCallback(() => {
    setChatKey((prev) => prev + 1);
  }, []);

  return (
    <div className="flex h-screen flex-col bg-gray-50">
      <Header
        hasApiKey={hasApiKey}
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
        currentKey={apiKey}
      />
    </div>
  );
}
