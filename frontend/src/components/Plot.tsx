"use client";

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

// Dynamically import Plotly to avoid SSR issues
const PlotlyPlot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[400px] items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
    </div>
  ),
});

interface PlotProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  className?: string;
}

export function Plot({
  data,
  layout = {},
  config = {},
  className = "",
}: PlotProps) {
  const defaultLayout: Partial<Plotly.Layout> = {
    autosize: true,
    margin: { l: 50, r: 50, t: 50, b: 50 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: {
      family: "system-ui, sans-serif",
      color: "#374151",
    },
    ...layout,
  };

  const defaultConfig: Partial<Plotly.Config> = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    ...config,
  };

  return (
    <div className={`w-full ${className}`}>
      <PlotlyPlot
        data={data}
        layout={defaultLayout}
        config={defaultConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}

// Helper function to parse plot data from API response
export function parsePlotFromResponse(content: string): {
  plotData: Plotly.Data[] | null;
  plotLayout: Partial<Plotly.Layout> | null;
} {
  // Look for JSON plot data in the response
  const plotMatch = content.match(/```json:plot\n([\s\S]*?)```/);

  if (plotMatch) {
    try {
      const plotJson = JSON.parse(plotMatch[1]);
      return {
        plotData: plotJson.data || null,
        plotLayout: plotJson.layout || null,
      };
    } catch {
      return { plotData: null, plotLayout: null };
    }
  }

  return { plotData: null, plotLayout: null };
}
