from typing import Dict, Any, List

import asyncio

from app.settings import settings
from app.db import get_cache_database_sync
from app.services import indian_api
from app.services.scraping import scrape_simple_news, crawl_with_scrapy, fetch_with_selenium

# LangChain / LangGraph
try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langgraph.graph import START, StateGraph  # type: ignore
except Exception:  # pragma: no cover
    ChatAnthropic = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    START = "start"  # type: ignore
    class StateGraph:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass
        def add_node(self, *_args, **_kwargs):
            pass
        def add_edge(self, *_args, **_kwargs):
            pass
        def compile(self):
            class _App:
                async def ainvoke(self, state):
                    return state
            return _App()


def _lc_model():
    if not ChatAnthropic:
        raise RuntimeError("LangChain/Anthropic not installed")
    return ChatAnthropic(model=settings.ANTHROPIC_MODEL, api_key=settings.ANTHROPIC_API_KEY, max_tokens=512)  # type: ignore


def _state() -> Dict[str, Any]:
    return {"symbol": None, "industry": "", "news": [], "scraped": [], "facts": [], "analysis": "", "plan": []}


async def _node_fetch_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch industry and API news for the company."""
    db = get_cache_database_sync()
    lname = str(state.get("symbol") or "").strip().lower()
    details = db.stock_details.find_one({"_norm_name": lname}) or {}
    industry = details.get("industry") or details.get("mgIndustry") or ""
    news_api = []
    try:
        news_api = await indian_api.fetch_news(symbol=state["symbol"], limit=10)
    except Exception:
        news_api = []
    state = dict(state)
    state["industry"] = industry
    state["news"] = news_api if isinstance(news_api, list) else []
    return state


async def _node_scrape_web(state: Dict[str, Any]) -> Dict[str, Any]:
    """Scrape a few public sources related to the company and industry."""
    company = state["symbol"]
    industry = state.get("industry", "")
    seeds: List[str] = [
        f"https://www.google.com/search?q={company}+news",
        f"https://www.google.com/search?q={company}+quarterly+results",
    ]
    if industry:
        seeds.append(f"https://www.google.com/search?q={industry}+industry+news+india")
    scraped = await scrape_simple_news(seeds)
    # augment with scrapy crawl results and one selenium fetch
    try:
        scraped.extend(crawl_with_scrapy(seeds[:2]))
    except Exception:
        pass
    try:
        scraped.append(fetch_with_selenium(seeds[0]))
    except Exception:
        pass
    state = dict(state)
    state["scraped"] = scraped
    return state


async def _node_analyze(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to synthesize key facts and a plan for what to fetch next."""
    model = _lc_model()
    if not ChatPromptTemplate:
        return state
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a diligent equity research assistant."),
        ("user", "Given company {symbol} in industry {industry}, combine API news and scraped snippets to extract 5-8 bullet facts and propose a brief plan of additional diligence items. Return JSON with keys facts (list) and plan (list).\nAPI News: {news}\nScraped: {scraped}"),
    ])
    chain = prompt | model
    try:
        res = await chain.ainvoke({"symbol": state["symbol"], "industry": state.get("industry", ""), "news": state.get("news", []), "scraped": state.get("scraped", [])})
        content = getattr(res, "content", "") or ""
    except Exception:
        content = ""
    import json
    facts: List[str] = []
    plan: List[str] = []
    try:
        parsed = json.loads(str(content))
        facts = parsed.get("facts") or []
        plan = parsed.get("plan") or []
    except Exception:
        facts = []
        plan = []
    state = dict(state)
    state["facts"] = facts
    state["plan"] = plan
    state["analysis"] = content
    return state


async def _node_peer_and_factors(state: Dict[str, Any]) -> Dict[str, Any]:
    """Gather peer metrics and produce structured AI factors to feed model."""
    db = get_cache_database_sync()
    lname = str(state.get("symbol") or "").strip().lower()
    details = db.stock_details.find_one({"_norm_name": lname}) or {}
    industry = details.get("industry") or details.get("mgIndustry") or ""
    # derive peer aggregates similar to predict_enriched
    peers = list(db.stock_details.find({"$or": [{"industry": industry}, {"mgIndustry": industry}]}).limit(30)) if industry else []
    peer_names = [str(p.get("_norm_name") or "") for p in peers if isinstance(p, dict)]
    # simple aggregation: count and avg risk
    risks = []
    for p in peers:
        rv = (p.get("riskMeter") or {})
        rv = rv.get("score") if isinstance(rv, dict) else rv
        try:
            risks.append(float(rv))
        except Exception:
            pass
    factors = {
        "peer_count": float(len(peer_names)),
        "peer_risk_mean": float(sum(risks) / len(risks)) if risks else 0.0,
    }
    # Ask LLM for forward-looking numeric factors from scraped/news
    model = _lc_model()
    if not ChatPromptTemplate:
        return state
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract numeric forward factors as JSON with keys: forward_eps_growth, demand_outlook, cost_pressure, regulatory_risk."),
        ("user", "Facts: {facts}\nScraped: {scraped}\nNews: {news}"),
    ])
    try:
        res = await (prompt | model).ainvoke({"facts": state.get("facts", []), "scraped": state.get("scraped", []), "news": state.get("news", [])})
        import json
        parsed = json.loads(getattr(res, "content", "") or "{}")
    except Exception:
        parsed = {}
    state = dict(state)
    state["peer_features"] = factors
    state["ai_factors"] = parsed
    return state


async def run_research_graph(symbol: str) -> Dict[str, Any]:
    """Execute a simple LangGraph pipeline for research."""
    graph = StateGraph(dict)
    graph.add_node("fetch", _node_fetch_context)
    graph.add_node("scrape", _node_scrape_web)
    graph.add_node("analyze", _node_analyze)
    graph.add_node("peers", _node_peer_and_factors)
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "scrape")
    graph.add_edge("scrape", "analyze")
    graph.add_edge("analyze", "peers")
    app = graph.compile()
    state = _state()
    state["symbol"] = symbol
    result = await app.ainvoke(state)
    # persist lite results in cache DB
    try:
        db = get_cache_database_sync()
        db.research_graph.update_one({"_norm_name": symbol.strip().lower(), "_kind": "graph"}, {"$set": {"symbol": symbol, "result": result}}, upsert=True)
    except Exception:
        pass
    return result


