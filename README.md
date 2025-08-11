## Investment Advisor – Full Web App

An AI‑powered market research and investment assistant with:
- Live NSE stock and AMFI mutual‑fund data
- Recommendation engine using Prophet and risk‑aware allocation
- AI insights (Claude via Anthropic) and research (to be upgraded with LangGraph agents)
- RESTful backend (FastAPI) and minimal dev UI
- MongoDB persistence for user profiles and portfolios

### Features
- Live quotes: stocks and mutual funds
- Recommendations: budget, horizon (1/3/5 yrs), risk tier; JSON allocations with perf summary
- Market insights: Claude‑powered concise analysis
- Research (beta): parallel web fetch + extraction
- MongoDB: user profile and saved portfolios

### Repo layout
```
app/                      # Streamlit app (unchanged)
backend/                  # FastAPI backend
  main.py                 # API entrypoint
  db.py                   # MongoDB connector (motor)
  schemas.py              # Pydantic models
  static/                 # minimal dev UI (served at /ui)
  services/
    market_data.py        # wraps src.live_data
    recommendation.py     # wraps src.recommendation_engine
    insights.py           # Anthropic Claude integration
    research.py           # web fetch + parse (BeautifulSoup)
    catalog.py            # IPOs/Bonds placeholders
    user.py               # Mongo-backed profiles/portfolios
src/                      # ML/data pipeline modules
```

### Requirements
See `requirements.txt`. Major libs: FastAPI, uvicorn, Pydantic, motor (MongoDB), anthropic, aiohttp, bs4, LangChain, LangGraph, Scrapy, Selenium, yfinance, Prophet, LightGBM, TensorFlow.

### Quickstart
1. Environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Environment variables
```
export ANTHROPIC_API_KEY=sk-ant-api03-...
export ANTHROPIC_MODEL=claude-3-haiku-20240307
export MONGODB_URI='mongodb+srv://<user>:<pass>@cluster0.c8ul7to.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
export MONGODB_DB=investment_advisor
```
3. Run API
```
uvicorn backend.main:app --host 127.0.0.1 --port 8010 --reload
```
Docs: `http://127.0.0.1:8010/docs` • Dev UI: `http://127.0.0.1:8010/ui/`

### API
- GET `/api/health`
- GET `/api/symbols?limit=50`
- GET `/api/mutuals?limit=50`
- GET `/api/quotes?tickers=RELIANCE,TCS`
- GET `/api/mfnv?schemes=118834,118835`
- POST `/api/recommendations`
```
{
  "budget": 100000,
  "horizon": 1,
  "risk": "Moderate",
  "products": ["Stocks","Mutual Funds"],
  "expected_cagr_min": 0.10,
  "max_risk_score": 1.2
}
```
Returns `{ portfolio: [...], perf: {...} }`.
- POST `/api/insights` → { query, tickers? }
- POST `/api/research` → { query, tickers?, competitors?, limit_sources? }
- GET `/api/ipos`, `/api/bonds` → placeholders
- Users: GET/POST `/api/users/{user_id}/profile`, POST `/api/users/{user_id}/portfolios`, GET `/api/users/{user_id}/portfolios`

### Data and Models
- Historical prices: `src/data_ingestion.fetch_stock`
- Features: `src/feature_engineering.full_pipeline`
- Models: `src/models/*` and `src/models/train_all.py`
- Ensemble scaffold: `src/models/ensemble.py`

### Design notes
- Prophet regressors at inference will be added via stored regressor metadata
- Risk proxy falls back to fundamentals beta if volatility unavailable
- In-memory caching for lists; consider Redis in production
- Mongo DB name controlled by `MONGODB_DB` – reuse same Atlas cluster

### Roadmap
- LangGraph multi-agent research with Claude 3 and citations
- Robust IPOs/Bonds ingestion and caching
- Better risk and correlation-aware allocation
- Next.js frontend with auth, charts, portfolio mgmt
- PDF/HTML reports
- Background jobs for retraining and refresh
- Observability and rate limiting

### Security
- Keep secrets in environment; never commit `.env`
- Validate inputs and add rate limiting for public endpoints


