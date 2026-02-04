"use client";

import { useState, useEffect } from "react";
import { Telescope, Github, ExternalLink, Key } from "lucide-react";
import { getCatalogueInfo } from "../lib/api";
import type { ApiTier } from "../lib/useApiKey";

interface HeaderProps {
  tier: ApiTier;
  maskedKey: string;
  onApiKeyClick: () => void;
  onLogoClick?: () => void;
}

export function Header({ tier, maskedKey, onApiKeyClick, onLogoClick }: HeaderProps) {
  const [catalogueInfo, setCatalogueInfo] = useState<{
    version: string;
    total_pulsars: number;
  } | null>(null);

  useEffect(() => {
    getCatalogueInfo()
      .then(setCatalogueInfo)
      .catch(() => {
        // Silently fail - will show placeholder
      });
  }, []);

  const tierDisplay = {
    "free-shared": { label: "Free Tier", color: "text-gray-500 hover:bg-gray-100 hover:text-gray-700" },
    openrouter: { label: maskedKey || "OpenRouter", color: "text-blue-600 hover:bg-blue-50 hover:text-blue-700" },
    anthropic: { label: maskedKey || "Anthropic", color: "text-green-600 hover:bg-green-50 hover:text-green-700" },
  }[tier];

  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
        {/* Logo and title */}
        <button
          onClick={onLogoClick}
          className="flex items-center gap-3 rounded-lg px-2 py-1 -ml-2 transition-colors hover:bg-gray-100"
          title="Return to home"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
            <Telescope className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">ATNF-Chat</h1>
          </div>
        </button>

        {/* Catalogue info and links */}
        <div className="flex items-center gap-4">
          {catalogueInfo && (
            <div className="hidden text-sm text-gray-500 sm:block">
              <span className="font-medium text-gray-700">
                {catalogueInfo.total_pulsars.toLocaleString()}
              </span>{" "}
              pulsars
              <span className="mx-2 text-gray-300">|</span>
              v{catalogueInfo.version}
            </div>
          )}

          <div className="flex items-center gap-2">
            {/* API Key / Tier button */}
            <button
              onClick={onApiKeyClick}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-colors ${tierDisplay.color}`}
            >
              <Key className="h-4 w-4" />
              <span className="hidden sm:inline">{tierDisplay.label}</span>
            </button>

            <a
              href="https://www.atnf.csiro.au/research/pulsar/psrcat/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-900"
            >
              ATNF Catalogue
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
            <a
              href="https://github.com/tkimpson/atnf-chat"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-900"
            >
              <Github className="h-4 w-4" />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}
