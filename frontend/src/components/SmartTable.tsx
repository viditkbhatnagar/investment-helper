import React, { useMemo } from "react";

type Props = {
  data: any[];
  maxRows?: number; // hard cap on how many rows we will ever render
  emptyMessage?: string;
  pageSize?: number; // rows per page
  maxCols?: number; // columns to infer at most
};

export default function SmartTable({ data, maxRows = 1000, emptyMessage = "No data", pageSize = 20, maxCols = 20 }: Props) {
  const rows: any[] = Array.isArray(data) ? data : [];
  const [page, setPage] = React.useState(1);

  const columns: string[] = useMemo(() => {
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const row of rows.slice(0, Math.min(maxRows, 200))) {
      if (row && typeof row === "object") {
        for (const key of Object.keys(row)) {
          if (!seen.has(key)) {
            seen.add(key);
            ordered.push(key);
            if (ordered.length >= maxCols) break;
          }
        }
      }
      if (ordered.length >= maxCols) break;
    }
    return ordered;
  }, [rows, maxRows, maxCols]);

  if (!rows || rows.length === 0) {
    return <div className="rounded border bg-white p-4 text-gray-600">{emptyMessage}</div>;
  }

  const toCell = (value: any) => {
    if (value === null || value === undefined) return "";
    if (typeof value === "object") return JSON.stringify(value);
    const s = String(value);
    return s.length > 120 ? s.slice(0, 117) + "…" : s;
  };

  const total = Math.min(rows.length, maxRows);
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const pageSafe = Math.min(Math.max(page, 1), totalPages);
  const start = (pageSafe - 1) * pageSize;
  const end = Math.min(start + pageSize, total);
  const pageRows = rows.slice(start, end);

  return (
    <div className="overflow-x-auto rounded border">
      <table className="min-w-full divide-y divide-gray-200 bg-white">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((c) => (
              <th key={c} className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {pageRows.map((row, i) => (
            <tr key={i} className="hover:bg-gray-50">
              {columns.map((c) => (
                <td key={c} className="px-3 py-2 text-sm text-gray-800 whitespace-nowrap">
                  {toCell(row?.[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center justify-between p-2 text-sm text-gray-600">
        <div>
          Showing {start + 1}-{end} of {total}
        </div>
        <div className="space-x-2">
          <button
            className="rounded border px-3 py-1 disabled:opacity-50"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={pageSafe <= 1}
          >
            Prev
          </button>
          <span>
            Page {pageSafe} / {totalPages}
          </span>
          <button
            className="rounded border px-3 py-1 disabled:opacity-50"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={pageSafe >= totalPages}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}


