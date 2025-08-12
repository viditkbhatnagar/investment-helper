import useSWR from "swr";
import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";
const fetcher = (url: string) => axios.get(`${API_BASE}${url}`).then((r) => r.data);

export function useApi<T = any>(url: string) {
  const { data, error, isLoading, mutate } = useSWR<T>(url, fetcher);
  return { data, error, isLoading, mutate };
}


