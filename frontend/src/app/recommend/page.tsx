"use client";
import { useEffect, useState } from "react";
import { getRecommendationsBySymbols, predictStock, researchStock, backtestStock, setBlendWeight, buildFeatures, trainModels, trainGlobalModels } from "@/lib/api";
import Plot from "@/components/Plot";
import Toast from "@/components/Toast";

export default function RecommendPage() {
  const [symbols, setSymbols] = useState("TCS,INFY,RELIANCE");
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("Eternal");
  const [pred, setPred] = useState<any | null>(null);
  const [research, setResearch] = useState<any | null>(null);
  const [bt, setBt] = useState<any | null>(null);
  const [alpha, setAlpha] = useState<string>("0.7");
  const [alphaH, setAlphaH] = useState<string>("5");
  const [days, setDays] = useState<string>("1,5,10");
  const [toast, setToast] = useState<{ msg: string; type?: "info" | "success" | "error" } | null>(null);

  const fetchRecs = async () => {
    setLoading(true);
    try {
      const res = await getRecommendationsBySymbols(symbols);
      setData(res);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchPred = async () => {
    setLoading(true);
    try {
      const horizon = days.split(",").map((d) => Number(d.trim())).filter((n) => !Number.isNaN(n) && n > 0);
      const res = await predictStock(name, horizon.length ? horizon : [1, 5, 10]);
      setPred(res);
    } finally {
      setLoading(false);
    }
  };

  const fetchResearch = async () => {
    setLoading(true);
    try {
      const horizon = days.split(",").map((d) => Number(d.trim())).filter((n) => !Number.isNaN(n) && n > 0);
      const res = await researchStock(name, horizon.length ? horizon : [1, 5, 10]);
      setResearch(res);
    } finally {
      setLoading(false);
    }
  };

  const runBacktest = async () => {
    setLoading(true);
    try {
      const res = await backtestStock(name, 5, 365);
      setBt(res);
      setToast({ msg: `Backtest done: MAE ${res?.mae ?? "-"}`, type: "success" });
    } finally {
      setLoading(false);
    }
  };

  const saveBlend = async () => {
    setLoading(true);
    try {
      const res = await setBlendWeight(name, Number(alphaH) || 5, Number(alpha));
      // refresh research to apply weight change on next call
      setResearch(null);
      setToast({ msg: `Saved alpha ${res?.alpha}`, type: "success" });
    } finally {
      setLoading(false);
    }
  };

  const runBuildFeatures = async () => {
    setLoading(true);
    try {
      const r = await buildFeatures(name);
      setToast({ msg: `Built features: ${r?.built ?? 0}`, type: "success" });
    } finally {
      setLoading(false);
    }
  };

  const runTrainModels = async () => {
    setLoading(true);
    try {
      const res = await trainModels(name);
      setToast({ msg: `Trained local: ${name}`, type: "success" });
    } finally {
      setLoading(false);
    }
  };

  const runTrainGlobalModels = async () => {
    setLoading(true);
    try {
      await trainGlobalModels();
      setToast({ msg: "Trained global models", type: "success" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Recommendations</h1>
      <div className="flex gap-2">
        <input
          className="w-full rounded border px-3 py-2"
          value={symbols}
          onChange={(e) => setSymbols(e.target.value)}
          placeholder="Enter symbols comma separated"
        />
        <button
          onClick={fetchRecs}
          className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
          disabled={loading}
        >
          {loading ? "Loading..." : "Get Recs"}
        </button>
      </div>
      <div className="rounded border bg-white p-3 space-y-2">
        <div className="font-medium">Predict</div>
        <div className="flex gap-2">
          <input
            className="w-full rounded border px-3 py-2"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Stock name (e.g., Eternal)"
          />
          <input
            className="w-40 rounded border px-3 py-2"
            value={days}
            onChange={(e) => setDays(e.target.value)}
            placeholder="Days (e.g., 1,5,10)"
          />
          <button
            onClick={fetchPred}
            className={`rounded bg-green-600 px-4 py-2 text-white disabled:opacity-50 ${loading ? "opacity-60" : ""}`}
            disabled={loading}
          >
            {loading ? "Loading..." : "Predict 1/5/10d"}
          </button>
          <button
            onClick={fetchResearch}
            className={`rounded bg-purple-600 px-4 py-2 text-white disabled:opacity-50 ${loading ? "opacity-60" : ""}`}
            disabled={loading}
          >
            {loading ? "Loading..." : "Research + Save"}
          </button>
          <button
            onClick={runBacktest}
            className={`rounded bg-amber-600 px-4 py-2 text-white disabled:opacity-50 ${loading ? "opacity-60" : ""}`}
            disabled={loading}
          >
            {loading ? "Loading..." : "Backtest 5d/365d"}
          </button>
          <button
            onClick={runBuildFeatures}
            className={`rounded bg-gray-700 px-4 py-2 text-white disabled:opacity-50 ${loading ? "opacity-60" : ""}`}
            disabled={loading}
          >
            {loading ? "Loading..." : "Build Features"}
          </button>
          <button
            onClick={runTrainModels}
            className={`rounded bg-gray-900 px-4 py-2 text-white disabled:opacity-50 ${loading ? "opacity-60" : ""}`}
            disabled={loading}
          >
            {loading ? "Loading..." : "Train Models"}
          </button>
          <button
            onClick={runTrainGlobalModels}
            className={`rounded bg-black px-4 py-2 text-white disabled:opacity-50 ${loading ? "opacity-60" : ""}`}
            disabled={loading}
          >
            {loading ? "Loading..." : "Train Global"}
          </button>
        </div>
        {pred && (
          <div className="space-y-2">
            <div className="text-sm">Today ({pred.today?.date}): Price {pred.today?.price ?? "-"}, Range {pred.today?.intraday_range ? `${pred.today.intraday_range[0]} - ${pred.today.intraday_range[1]}` : "-"}</div>
            <div className="text-sm font-medium">Predictions</div>
            <table className="min-w-full divide-y divide-gray-200 bg-white text-sm">
              <tbody className="divide-y divide-gray-100">
                {Object.entries(pred.predictions || {}).map(([date, v]: any) => (
                  <tr key={date}>
                    <td className="px-3 py-2 w-36">{date}</td>
                    <td className="px-3 py-2">{v?.point?.toFixed ? v.point.toFixed(2) : v?.point ?? "-"}</td>
                    <td className="px-3 py-2">{Array.isArray(v?.range) ? `${v.range[0]} - ${v.range[1]}` : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {/* Quantile bands chart for predictions if available in metrics */}
            {(() => {
              try {
                const entries = Object.entries(pred.predictions || {});
                const dates = entries.map(([d]) => d);
                const mid = entries.map(([, v]: any) => (v?.point ?? null));
                const q = pred.metrics || {};
                const diffDays = (d: string) => Math.round((new Date(d).getTime() - new Date(pred.today?.date || d).getTime()) / (1000*60*60*24));
                const low = dates.map((d: any) => (q?.[String(diffDays(d))]?.quantiles?.q20 ?? null));
                const high = dates.map((d: any) => (q?.[String(diffDays(d))]?.quantiles?.q80 ?? null));
                if (dates.length) {
                  return <Plot x={dates} y={mid} title="Predicted (P50) with Bands" bands={{ low, mid, high }} />;
                }
              } catch (e) {}
              return null;
            })()}
            {pred.metrics && (
              <div className="space-y-1 text-xs text-gray-700">
                <div className="font-medium">Training metrics</div>
                <ul className="list-disc pl-5">
                  {Object.entries(pred.metrics).map(([h, m]: any) => (
                    <li key={h}>H{h}: MAE {m?.mae ?? "-"} | MAPE {m?.mape ?? "-"} | CV-MAE {m?.cv_mae ?? "-"} | n {m?.n_samples ?? "-"}</li>
                  ))}
                </ul>
              </div>
            )}
            {Array.isArray(pred.drivers) && pred.drivers.length > 0 && (
              <div className="space-y-2 text-xs text-gray-700">
                <div className="text-sm font-medium">Top drivers (P50 SHAP)</div>
                {/* Bar chart using Plot component */}
                {(() => {
                  try {
                    const xs = pred.drivers.map((d: any) => d.feature);
                    const ys = pred.drivers.map((d: any) => d.importance);
                    return <Plot x={xs} y={ys} title="Drivers" />;
                  } catch (e) { return null; }
                })()}
              </div>
            )}
            <div className="text-sm font-medium">Explain</div>
            <pre className="overflow-auto text-xs text-gray-800">{JSON.stringify(pred.explain, null, 2)}</pre>
            {pred.report && typeof pred.report === "string" && (
              <div className="rounded border bg-gray-50 p-2 text-sm whitespace-pre-wrap">{pred.report}</div>
            )}
            {/* Real-time accuracy (last backtest MAE if available via backtest fetch) */}
            {bt?.mae !== undefined && (
              <div className="text-sm">Real-time model MAE (backtest window): {bt.mae}</div>
            )}
          </div>
        )}
      </div>
      {research && (
        <div className="rounded border bg-white p-3 space-y-2">
          <div className="font-medium">Research Snapshot</div>
          <div className="rounded border p-2 text-sm space-y-2">
            <div className="font-medium">Blend weight</div>
            <div className="flex gap-2 items-center">
              <label className="text-xs text-gray-600">Horizon</label>
              <input className="w-16 rounded border px-2 py-1" value={alphaH} onChange={(e) => setAlphaH(e.target.value)} />
              <label className="text-xs text-gray-600">Alpha (0..1)</label>
              <input className="w-24 rounded border px-2 py-1" value={alpha} onChange={(e) => setAlpha(e.target.value)} />
              <button onClick={saveBlend} className="rounded bg-blue-600 px-3 py-1 text-white disabled:opacity-50" disabled={loading}>Save</button>
            </div>
            <div className="text-xs text-gray-600">Alpha is weight for model vs AI. 1 = model only.</div>
          </div>
          <div className="text-sm">AI Notes</div>
          {research.ai_notes && (
            <div className="rounded border bg-gray-50 p-2 text-sm whitespace-pre-wrap">{research.ai_notes}</div>
          )}
          <div className="text-sm font-medium">Blended Predictions</div>
          <table className="min-w-full divide-y divide-gray-200 bg-white text-sm">
            <tbody className="divide-y divide-gray-100">
              {Object.entries(research.predictions_blended || {}).map(([date, v]: any) => (
                <tr key={date}>
                  <td className="px-3 py-2 w-36">{date}</td>
                  <td className="px-3 py-2">{v?.point?.toFixed ? v.point.toFixed(2) : v?.point ?? "-"}</td>
                  <td className="px-3 py-2">{Array.isArray(v?.range) ? `${v.range[0]} - ${v.range[1]}` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-sm font-medium">Today's News</div>
          <ul className="list-disc pl-6 text-sm">
            {(research.todays_news || []).map((n: any, idx: number) => (
              <li key={idx}>{n?.title}</li>
            ))}
          </ul>
        </div>
      )}
      {bt && (
        <div className="rounded border bg-white p-3 space-y-2">
          <div className="font-medium">Backtest (Expanding window)</div>
          <div className="text-sm">MAE: {bt.mae} | N: {bt.n} | Horizon: {bt.horizon} | Window: {bt.window_days}d</div>
          <div className="text-xs text-gray-700">Series size: {bt.series?.date?.length || 0}</div>
          {bt.series?.date?.length ? (
            <Plot x={bt.series.date} y={bt.series.y_true} title="Backtest (True vs P50 with Bands)" bands={{ mid: bt.series.y_pred, low: bt.quantile_series?.q20, high: bt.quantile_series?.q80 }} />
          ) : null}
        </div>
      )}
      {/* Research Graph Panel */}
      {research && (
        <div className="rounded border bg-white p-3 space-y-2">
          <div className="font-medium">Research Graph</div>
          {Array.isArray(research.todays_news) && research.todays_news.length > 0 && (
            <div className="text-sm">
              <div className="font-medium">Data sources used</div>
              <ul className="list-disc pl-5">
                <li>intel_daily (signals, facts)</li>
                <li>predictions_daily (model outputs)</li>
                <li>features_daily (technical + fundamentals + peers + AI)</li>
                <li>recent_announcements / news (sentiment)</li>
                <li>stock_details (riskMeter, industry)</li>
                <li>fiftytwo_week, price_shockers, corporate_actions</li>
              </ul>
            </div>
          )}
        </div>
      )}
      {toast && <Toast message={toast.msg} type={toast.type} onClose={() => setToast(null)} />}
      <ul className="space-y-2">
        {data.map((r, idx) => (
          <li key={idx} className="rounded border bg-white p-3">
            <div className="font-medium">{r.symbol}</div>
            <div className="text-sm text-gray-600">Rating: {r.rating}</div>
            {r.rationale && (
              <div className="text-sm text-gray-600">{r.rationale}</div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}


