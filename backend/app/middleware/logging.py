import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        # Log to our own channel to avoid uvicorn.access' AccessFormatter (expects 5 args)
        logging.getLogger("app.request").info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            str(response.status_code),
            float(duration_ms),
        )
        return response


