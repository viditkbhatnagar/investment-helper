"use client";
import React, { useEffect } from "react";

export default function Toast({ message, type = "info", onClose, duration = 3000 }: { message: string; type?: "info" | "success" | "error"; onClose: () => void; duration?: number }) {
  useEffect(() => {
    const id = setTimeout(() => onClose(), duration);
    return () => clearTimeout(id);
  }, [onClose, duration]);
  const color = type === "success" ? "bg-green-600" : type === "error" ? "bg-red-600" : "bg-gray-800";
  return (
    <div className={`fixed right-4 top-4 z-50 rounded px-4 py-2 text-white shadow ${color}`}
      role="status" aria-live="polite">
      {message}
    </div>
  );
}


