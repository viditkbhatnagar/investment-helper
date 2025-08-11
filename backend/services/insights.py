from __future__ import annotations

import os
import asyncio
from typing import List

from backend.schemas import InsightRequest, InsightResponse


async def generate_market_insights(req: InsightRequest) -> InsightResponse:
    """
    Generate a brief market insight using Anthropic Claude if ANTHROPIC_API_KEY is set.
    Non-blocking via asyncio.to_thread to avoid stalling the event loop.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return InsightResponse(
            insight=(
                "LLM insights service not configured. Set ANTHROPIC_API_KEY to enable. "
                f"Query received: '{req.query}'."
            ),
            sources=[],
        )

    def _call_anthropic() -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        system_prompt = (
            "You are an AI Market Research Analyst. \n"
            "Produce concise, actionable insights with numbered bullets. \n"
            "Include brief rationale and potential risks. Avoid speculation."
        )
        user_prompt = req.query
        if req.tickers:
            user_prompt += f"\n\nFocus tickers: {', '.join(req.tickers)}"

        msg = client.messages.create(
            model=model,
            max_tokens=900,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Anthropic SDK returns list of content blocks; join text parts
        parts = []
        for block in msg.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    insight_text = await asyncio.to_thread(_call_anthropic)
    return InsightResponse(insight=insight_text, sources=[])


