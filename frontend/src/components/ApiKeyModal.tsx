"use client";

import { useState, useEffect } from "react";
import { X, ExternalLink, Eye, EyeOff, Telescope } from "lucide-react";
import type { ApiTier } from "../lib/useApiKey";

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (apiKey: string) => void;
  onTierSelect: (tier: ApiTier) => void;
  currentKey: string;
  currentTier: ApiTier;
}

export function ApiKeyModal({
  isOpen,
  onClose,
  onSave,
  onTierSelect,
  currentKey,
  currentTier,
}: ApiKeyModalProps) {
  const [selectedTier, setSelectedTier] = useState<ApiTier>(currentTier);
  const [openrouterKey, setOpenrouterKey] = useState("");
  const [anthropicKey, setAnthropicKey] = useState("");
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    setSelectedTier(currentTier);
    if (currentTier === "openrouter" && currentKey) {
      setOpenrouterKey(currentKey);
    } else if (currentTier === "anthropic" && currentKey) {
      setAnthropicKey(currentKey);
    }
  }, [currentTier, currentKey, isOpen]);

  if (!isOpen) return null;

  const handleContinue = () => {
    if (selectedTier === "free-shared") {
      onSave("");
      onTierSelect("free-shared");
    } else if (selectedTier === "openrouter") {
      onSave(openrouterKey.trim());
    } else {
      onSave(anthropicKey.trim());
    }
    onClose();
  };

  const handleDismiss = () => {
    onTierSelect("free-shared");
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      handleDismiss();
    }
    if (e.key === "Enter" && canContinue) {
      handleContinue();
    }
  };

  const activeKey = selectedTier === "openrouter" ? openrouterKey : anthropicKey;
  const canContinue =
    selectedTier === "free-shared" || activeKey.trim().length > 0;

  const buttonLabel =
    selectedTier === "free-shared" ? "Start Chatting" : "Save Key & Start";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={handleDismiss}
      />

      {/* Modal */}
      <div className="relative w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
              <Telescope className="h-5 w-5 text-white" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">
              Welcome to ATNF-Chat
            </h2>
          </div>
          <button
            onClick={handleDismiss}
            className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <p className="mb-5 text-sm text-gray-600">
          Choose how you&apos;d like to connect. You can change this anytime from
          the header.
        </p>

        {/* Tier options */}
        <div className="mb-5 space-y-3">
          {/* Free Shared */}
          <label
            className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors ${
              selectedTier === "free-shared"
                ? "border-blue-500 bg-blue-50"
                : "border-gray-200 hover:border-gray-300"
            }`}
          >
            <input
              type="radio"
              name="tier"
              value="free-shared"
              checked={selectedTier === "free-shared"}
              onChange={() => setSelectedTier("free-shared")}
              className="mt-0.5 h-4 w-4 text-blue-600"
            />
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">
                Free Shared Tier
              </div>
              <p className="mt-0.5 text-xs text-gray-500">
                Works immediately &mdash; no setup needed. Shared key with
                limited capacity; may degrade under heavy use.
              </p>
            </div>
          </label>

          {/* OpenRouter */}
          <label
            className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors ${
              selectedTier === "openrouter"
                ? "border-blue-500 bg-blue-50"
                : "border-gray-200 hover:border-gray-300"
            }`}
          >
            <input
              type="radio"
              name="tier"
              value="openrouter"
              checked={selectedTier === "openrouter"}
              onChange={() => setSelectedTier("openrouter")}
              className="mt-0.5 h-4 w-4 text-blue-600"
            />
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">
                Your OpenRouter Key{" "}
                <span className="font-normal text-gray-500">(free)</span>
              </div>
              <p className="mt-0.5 text-xs text-gray-500">
                Get your own free key for personal rate limits &mdash; takes 30
                seconds.
              </p>
              {selectedTier === "openrouter" && (
                <div className="mt-2">
                  <div className="relative">
                    <input
                      type={showKey ? "text" : "password"}
                      value={openrouterKey}
                      onChange={(e) => setOpenrouterKey(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="sk-or-..."
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 pr-10 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showKey ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                  <a
                    href="https://openrouter.ai/settings/keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1.5 inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                  >
                    Get a free key from OpenRouter
                    <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              )}
            </div>
          </label>

          {/* Anthropic */}
          <label
            className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors ${
              selectedTier === "anthropic"
                ? "border-blue-500 bg-blue-50"
                : "border-gray-200 hover:border-gray-300"
            }`}
          >
            <input
              type="radio"
              name="tier"
              value="anthropic"
              checked={selectedTier === "anthropic"}
              onChange={() => setSelectedTier("anthropic")}
              className="mt-0.5 h-4 w-4 text-blue-600"
            />
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">
                Your Anthropic Key{" "}
                <span className="font-normal text-gray-500">(paid)</span>
              </div>
              <p className="mt-0.5 text-xs text-gray-500">
                Best quality. Uses Claude directly with your own account.
              </p>
              {selectedTier === "anthropic" && (
                <div className="mt-2">
                  <div className="relative">
                    <input
                      type={showKey ? "text" : "password"}
                      value={anthropicKey}
                      onChange={(e) => setAnthropicKey(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="sk-ant-..."
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 pr-10 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showKey ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                  <a
                    href="https://console.anthropic.com/settings/keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1.5 inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                  >
                    Get an API key from Anthropic
                    <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              )}
            </div>
          </label>
        </div>

        {/* Actions */}
        <button
          onClick={handleContinue}
          disabled={!canContinue}
          className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {buttonLabel}
        </button>

        {/* Privacy note */}
        <p className="mt-4 text-center text-xs text-gray-500">
          Your API key is stored only in your browser&apos;s local storage.
          Keys are sent to the server only to authenticate API requests.
        </p>
      </div>
    </div>
  );
}
