"use client";
import { useEffect, useState } from "react";
import TabsContainer from "@/components/TabsContainer";
import SmartTable from "@/components/SmartTable";
import { getMarketStocks, getMarketMutualFunds, getMarketIpos, getMarketCommodities } from "@/lib/api";

export default function HomePage() {
  const [stocks, setStocks] = useState<any>({ trending: [], bse_most_active: [], nse_most_active: [] });
  const [funds, setFunds] = useState<any[]>([]);
  const [ipos, setIpos] = useState<any[]>([]);
  const [commodities, setCommodities] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [s, mf, ip, cm] = await Promise.all([
          getMarketStocks().catch(() => ({ trending: [], bse_most_active: [], nse_most_active: [] })),
          getMarketMutualFunds().catch(() => []),
          getMarketIpos().catch(() => []),
          getMarketCommodities().catch(() => []),
        ]);
        setStocks(s);
        setFunds(mf);
        setIpos(ip);
        setCommodities(cm);
      } catch (e: any) {
        setError(e?.message || "Failed to load data");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const Table = ({ data }: { data: any[] }) => <SmartTable data={data} maxRows={1000} />;

  const stocksTab = (
    <div className="space-y-6">
      <div>
        <h2 className="mb-2 font-semibold">Trending</h2>
        <Table data={stocks.trending} />
      </div>
      <div>
        <h2 className="mb-2 font-semibold">BSE Most Active</h2>
        <Table data={stocks.bse_most_active} />
      </div>
      <div>
        <h2 className="mb-2 font-semibold">NSE Most Active</h2>
        <Table data={stocks.nse_most_active} />
      </div>
    </div>
  );

  const tabs = [
    { id: "stocks", title: "Stocks", content: stocksTab },
    { id: "mutuals", title: "Mutual Funds", content: <Table data={funds} /> },
    { id: "ipos", title: "IPOs", content: <Table data={ipos} /> },
    { id: "commodities", title: "Commodities", content: <Table data={commodities} /> },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Investment Advisor</h1>
      {loading ? (
        <div>Loading...</div>
      ) : error ? (
        <div className="rounded border border-red-300 bg-red-50 p-3 text-red-800">{error}</div>
      ) : (
        <TabsContainer tabs={tabs} />
      )}
    </div>
  );
}


