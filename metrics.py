import time
from prometheus_client import Histogram, Counter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency by method and path and status",
    ["method", "path", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10),
)

RECOMMEND_COUNT = Counter(
    "api_recommend_total",
    "Total number of recommendations served",
)

RECOMMEND_SCORE = Histogram(
    "api_recommend_score_bucket",
    "Histogram of recommendation scores",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

class RequestLatencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        # Option : regrouper les chemins dynamiques (ex: /games/123 -> /games/{id})
        if path.count("/") > 2:
            parts = path.split("/")
            parts[-1] = "{param}"
            path = "/".join(parts)
        REQUEST_LATENCY.labels(request.method, path, str(response.status_code)).observe(elapsed)
        return response

# Helpers à appeler dans l’endpoint reco
def observe_recommendation(scores):
    RECOMMEND_COUNT.inc()
    for s in scores:
        RECOMMEND_SCORE.observe(float(s))
