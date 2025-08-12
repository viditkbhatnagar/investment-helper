import "./globals.css";
import { ReactNode } from "react";
import Link from "next/link";

export const metadata = {
  title: "Investment Advisor",
  description: "Discover insights and recommendations",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900">
        <header className="border-b bg-white">
          <div className="mx-auto flex max-w-5xl items-center justify-between p-4">
            <div className="font-semibold">Investment Advisor</div>
            <nav className="flex gap-4 text-sm">
              <Link className="text-blue-600 hover:underline" href="/">Home</Link>
              <Link className="text-blue-600 hover:underline" href="/recommend">Recommendations</Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-5xl p-4">{children}</main>
      </body>
    </html>
  );
}


