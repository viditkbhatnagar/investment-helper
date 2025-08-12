import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";

const client = axios.create({ baseURL: `${API_BASE}/api` });

export async function getRecommendationsBySymbols(symbolsCsv: string) {
  const { data } = await client.get(`/recommendations`, { params: { symbols: symbolsCsv } });
  return data;
}

export async function getInsights() {
  const { data } = await client.get(`/insights/`);
  return data;
}

export async function getMe() {
  const { data } = await client.get(`/users/me`);
  return data;
}

export async function getQuotes(symbols: string[]) {
  const { data } = await client.get(`/stocks/quotes`, {
    params: { symbols: symbols.join(",") },
  });
  return data;
}

// Market data wrappers
export async function getMarketStocks() {
  const { data } = await client.get(`/market/stocks`);
  return data;
}

export async function getMarketMutualFunds() {
  const { data } = await client.get(`/market/mutual-funds`);
  return data;
}

export async function getMarketIpos() {
  const { data } = await client.get(`/market/ipos`);
  return data;
}

export async function getMarketCommodities() {
  const { data } = await client.get(`/market/commodities`);
  return data;
}

export async function predictStock(name: string, days: number[]) {
  const { data } = await client.get(`/recommendations/predict`, {
    params: { name, days: days.join(",") },
  });
  return data;
}

export async function researchStock(name: string, days: number[]) {
  const { data } = await client.get(`/recommendations/research`, {
    params: { name, days: days.join(",") },
  });
  return data;
}

export async function backtestStock(name: string, horizon: number, windowDays: number) {
  const { data } = await client.get(`/market/backtest`, {
    params: { name, horizon, window_days: windowDays },
  });
  return data;
}

export async function setBlendWeight(name: string, horizon: number, alpha: number) {
  const { data } = await client.post(`/recommendations/blend-weight`, { alpha }, {
    params: { name, horizon },
  });
  return data;
}

export async function buildFeatures(name: string) {
  const { data } = await client.post(`/market/features`, null, { params: { name } });
  return data;
}

export async function trainModels(name: string) {
  const { data } = await client.post(`/market/train`, null, { params: { name } });
  return data;
}

export async function trainGlobalModels() {
  const { data } = await client.post(`/market/train-global`);
  return data;
}


