"use client";
import dynamic from "next/dynamic";
import React from "react";

const Plotly = dynamic(() => import("react-plotly.js"), { ssr: false } as any) as any;

export default function Plot({ x = [], y = [], title = "", bands }: { x: any[]; y: number[]; title?: string; bands?: { low?: number[]; mid?: number[]; high?: number[] } }) {
  return (
    <div className="rounded border p-2">
      <Plotly
        data={[
          { x, y, name: "value", type: "scatter", mode: "lines", line: { color: "#2563eb" } },
          ...(bands?.low ? [{ x, y: bands.low, name: "P20", type: "scatter", mode: "lines", line: { color: "#ef4444", dash: "dot" } }] : []),
          ...(bands?.mid ? [{ x, y: bands.mid, name: "P50", type: "scatter", mode: "lines", line: { color: "#10b981", dash: "dash" } }] : []),
          ...(bands?.high ? [{ x, y: bands.high, name: "P80", type: "scatter", mode: "lines", line: { color: "#f59e0b", dash: "dot" } }] : []),
        ] as any}
        layout={{ title, autosize: true, margin: { l: 40, r: 20, t: 30, b: 30 }, legend: { orientation: "h" } }}
        style={{ width: "100%", height: "320px" }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}


