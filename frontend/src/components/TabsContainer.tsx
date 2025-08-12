import React, { useState } from "react";

export default function TabsContainer({
  tabs,
}: {
  tabs: { id: string; title: string; content: React.ReactNode }[];
}) {
  const [active, setActive] = useState(tabs[0]?.id);

  return (
    <div>
      <div className="mb-3 flex gap-2 overflow-x-auto border-b">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${
              active === t.id ? "border-blue-600 text-blue-600" : "border-transparent"
            }`}
          >
            {t.title}
          </button>
        ))}
      </div>
      <div>
        {tabs.map((t) => (
          <div key={t.id} className={active === t.id ? "block" : "hidden"}>
            {t.content}
          </div>
        ))}
      </div>
    </div>
  );
}


