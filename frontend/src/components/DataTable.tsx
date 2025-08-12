import React from "react";

type Column<T> = {
  key: keyof T;
  header: string;
};

export function DataTable<T extends Record<string, any>>({
  data,
  columns,
}: {
  data: T[];
  columns: Column<T>[];
}) {
  return (
    <div className="overflow-x-auto rounded border">
      <table className="min-w-full divide-y divide-gray-200 bg-white">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={String(c.key)} className="px-4 py-2 text-left text-sm font-semibold">
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {data.map((row, idx) => (
            <tr key={idx}>
              {columns.map((c) => (
                <td key={String(c.key)} className="px-4 py-2 text-sm text-gray-700">
                  {String(row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


