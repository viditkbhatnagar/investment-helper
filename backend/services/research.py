from __future__ import annotations

from typing import List

import asyncio
import aiohttp
from bs4 import BeautifulSoup

from backend.schemas import ResearchRequest, ResearchResponse


async def _fetch(session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
    try:
        async with session.get(url, timeout=15) as resp:
            text = await resp.text()
            return url, text
    except Exception:
        return url, ""


def _extract_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())[:20000]


async def run_research(req: ResearchRequest) -> ResearchResponse:
    # Very light parallel fetcher for a few generic news/finance pages
    base_queries = [req.query]
    if req.tickers:
        base_queries += req.tickers
    urls = []
    for q in base_queries:
        qx = q.replace(" ", "+")
        urls.append(f"https://www.google.com/search?q={qx}+site:moneycontrol.com")
        urls.append(f"https://www.google.com/search?q={qx}+site:economictimes.indiatimes.com")
    urls = urls[: req.limit_sources]

    pages: List[tuple[str, str]] = []
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        results = await asyncio.gather(*[_fetch(session, u) for u in urls])
        pages.extend(results)

    texts = [_extract_text(html) for _, html in pages]
    # Summarization will be moved to Anthropic in next step; simple concat for now
    snippet = "\n\n".join(t[:1000] for t in texts if t)[:4000]
    sources = [u for u, _ in pages]
    return ResearchResponse(insight=snippet or "No content fetched.", sources=sources)


