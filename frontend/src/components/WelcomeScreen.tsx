"use client";

import { Telescope, Search, BarChart3, Code } from "lucide-react";

interface WelcomeScreenProps {
  onExampleClick: (example: string) => void;
}

const EXAMPLE_QUERIES = [
  {
    icon: Search,
    title: "Find pulsars",
    query: "Show me all millisecond pulsars in globular clusters",
  },
  {
    icon: Telescope,
    title: "Pulsar details",
    query: "Tell me about the Vela pulsar",
  },
  {
    icon: BarChart3,
    title: "Statistics",
    query: "What's the distribution of pulsar periods?",
  },
  {
    icon: Code,
    title: "Export code",
    query: "Find binary pulsars with orbital periods less than 1 day and show me the Python code",
  },
];

export function WelcomeScreen({ onExampleClick }: WelcomeScreenProps) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-4 py-12">
      <div className="max-w-2xl text-center">
        {/* Logo/Title */}
        <div className="mb-8">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg">
            <Telescope className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900">ATNF-Chat</h1>
          <p className="mt-2 text-lg text-gray-600">
            Natural language interface to the ATNF Pulsar Catalogue
          </p>
        </div>

        {/* Description */}
        <div className="mb-8 rounded-xl bg-blue-50 p-6 text-left">
          <h2 className="mb-3 font-semibold text-blue-900">What can I help you with?</h2>
          <ul className="space-y-2 text-sm text-blue-800">
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
              Query the catalogue using natural language
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
              Look up specific pulsars by name (J2000 or B1950)
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
              Compute derived parameters (magnetic field, age, spin-down luminosity)
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
              Generate statistics and correlations
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
              Export reproducible Python code
            </li>
          </ul>
        </div>

        {/* Example queries */}
        <div>
          <h3 className="mb-4 text-sm font-medium text-gray-500">Try an example</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {EXAMPLE_QUERIES.map((example) => (
              <button
                key={example.title}
                onClick={() => onExampleClick(example.query)}
                className="group flex items-start gap-3 rounded-xl border border-gray-200 bg-white p-4 text-left transition-all hover:border-blue-300 hover:shadow-md"
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-gray-600 transition-colors group-hover:bg-blue-100 group-hover:text-blue-600">
                  <example.icon className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <div className="font-medium text-gray-900">{example.title}</div>
                  <div className="mt-0.5 text-sm text-gray-500 line-clamp-2">
                    {example.query}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
