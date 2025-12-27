"use client";

import { useState, useEffect } from "react";
import { X, Key, ExternalLink, Eye, EyeOff, Zap } from "lucide-react";

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (apiKey: string) => void;
  currentKey: string;
  isOptional?: boolean;
  rateLimitPerMinute?: number;
}

export function ApiKeyModal({
  isOpen,
  onClose,
  onSave,
  currentKey,
  isOptional = false,
  rateLimitPerMinute = 0,
}: ApiKeyModalProps) {
  const [apiKey, setApiKey] = useState(currentKey);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    setApiKey(currentKey);
  }, [currentKey, isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(apiKey.trim());
    onClose();
  };

  const handleSkip = () => {
    onSave(""); // Clear any existing key
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (apiKey.trim() || isOptional)) {
      handleSave();
    } else if (e.key === "Escape") {
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-md rounded-xl bg-white p-6 shadow-2xl">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100">
              <Key className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">API Key</h2>
              {isOptional && (
                <span className="text-xs font-medium text-green-600">Optional</span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Description */}
        {isOptional ? (
          <div className="mb-4 space-y-3">
            <div className="flex items-start gap-2 rounded-lg bg-green-50 p-3">
              <Zap className="mt-0.5 h-4 w-4 flex-shrink-0 text-green-600" />
              <p className="text-sm text-green-800">
                <strong>Ready to use!</strong> This app has a built-in API key, so you can
                start chatting immediately without providing your own.
              </p>
            </div>
            <p className="text-sm text-gray-600">
              Want to use your own API key? Enter it below for higher rate limits
              and to use your own Anthropic account.
              {rateLimitPerMinute > 0 && (
                <span className="block mt-1 text-xs text-gray-500">
                  Server limit: {rateLimitPerMinute} requests/minute
                </span>
              )}
            </p>
          </div>
        ) : (
          <p className="mb-4 text-sm text-gray-600">
            Enter your Anthropic API key to use ATNF-Chat. Your key is stored
            locally in your browser and sent directly to Anthropic.
          </p>
        )}

        {/* Input */}
        <div className="mb-4">
          <label
            htmlFor="api-key"
            className="mb-1.5 block text-sm font-medium text-gray-700"
          >
            Anthropic API Key {isOptional && <span className="text-gray-400">(optional)</span>}
          </label>
          <div className="relative">
            <input
              id="api-key"
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="sk-ant-..."
              className="w-full rounded-lg border border-gray-300 px-4 py-2.5 pr-10 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              autoFocus={!isOptional}
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
        </div>

        {/* Help link */}
        <a
          href="https://console.anthropic.com/settings/keys"
          target="_blank"
          rel="noopener noreferrer"
          className="mb-6 inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700"
        >
          Get an API key from Anthropic
          <ExternalLink className="h-3.5 w-3.5" />
        </a>

        {/* Actions */}
        <div className="flex gap-3">
          {isOptional ? (
            <>
              <button
                onClick={handleSkip}
                className="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Use Server Key
              </button>
              <button
                onClick={handleSave}
                disabled={!apiKey.trim()}
                className="flex-1 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Use My Key
              </button>
            </>
          ) : (
            <>
              <button
                onClick={onClose}
                className="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={!apiKey.trim()}
                className="flex-1 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Save Key
              </button>
            </>
          )}
        </div>

        {/* Privacy note */}
        <p className="mt-4 text-center text-xs text-gray-500">
          {isOptional
            ? "Your API key (if provided) is stored only in your browser."
            : "Your API key is stored only in your browser's local storage."}
        </p>
      </div>
    </div>
  );
}
