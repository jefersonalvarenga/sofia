"""
Tests for SecurityMiddleware proxy-aware client IP resolution and public path bypass.

Bug context: in production (Easypanel), `request.client.host` returns the IP of the
internal reverse proxy (e.g. `10.11.0.17`), not the real client. When bots scanned
suspicious paths through the proxy, the middleware banned the proxy IP and then
rejected every subsequent request — including legitimate Evolution webhook deliveries
that ride through the same proxy. The fix has two parts:

  1. `_get_client_ip` resolves the real client from X-Forwarded-For / X-Real-IP /
     `request.client.host` (in that order).
  2. Public paths (`/v1/iris/webhook/*` and `/v1/health`) skip the suspicious-path /
     extension / user-agent / blocked-IP checks entirely.
"""

from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from app.core.security import SecurityMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityMiddleware, rate_limit=1000)

    @app.get("/v1/iris/webhook/evolution")
    @app.post("/v1/iris/webhook/evolution")
    async def webhook_evolution():
        return JSONResponse({"ok": True})

    @app.get("/v1/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/v1/anything")
    async def anything():
        return JSONResponse({"ok": True})

    return app


def _get_middleware(client: TestClient) -> SecurityMiddleware:
    """Walk the Starlette middleware stack to grab the SecurityMiddleware instance."""
    app = client.app.middleware_stack
    while app is not None:
        if isinstance(app, SecurityMiddleware):
            return app
        app = getattr(app, "app", None)
    raise RuntimeError("SecurityMiddleware not found in stack")


class TestProxyAwareClientIP:
    def test_x_forwarded_for_first_ip_is_used_for_blocking(self):
        """RED 1: real client IP (first in XFF chain) is the one banned, not the proxy."""
        app = _build_app()
        client = TestClient(app)

        # Trigger a suspicious-path block. The TestClient default `client.host` is
        # `testclient` (acts as the "proxy"); the real client is 1.2.3.4.
        resp = client.get(
            "/.env.bak",
            headers={"X-Forwarded-For": "1.2.3.4, 10.11.0.17"},
        )
        assert resp.status_code == 403

        middleware = _get_middleware(client)
        # The REAL client should be blocked, not the proxy.
        assert "1.2.3.4" in middleware.blocked_ips
        assert "10.11.0.17" not in middleware.blocked_ips
        assert "testclient" not in middleware.blocked_ips


class TestPublicWebhookBypass:
    def test_webhook_passes_even_when_proxy_ip_is_blocked(self):
        """RED 2: webhook endpoint must succeed even if `request.client.host` is in blocked_ips."""
        app = _build_app()
        client = TestClient(app)

        middleware = _get_middleware(client)
        # Pretend the proxy IP got banned in a previous request.
        middleware.blocked_ips["testclient"] = datetime.now() + timedelta(minutes=30)

        resp = client.post(
            "/v1/iris/webhook/evolution",
            json={},
        )
        assert resp.status_code == 200, (
            f"Webhook must bypass the blocked-IP check; got {resp.status_code}"
        )

    def test_webhook_with_suspicious_query_string_still_passes(self):
        """RED 3: query string noise must not turn a webhook into a 'suspicious path' match."""
        app = _build_app()
        client = TestClient(app)

        # `?param=.env.bak` should be ignored — the request path is the webhook.
        resp = client.post(
            "/v1/iris/webhook/evolution?param=.env.bak",
            json={},
        )
        assert resp.status_code == 200


class TestSuspiciousPathStillBlocksRealClient:
    def test_suspicious_path_blocks_real_client_not_proxy(self):
        """RED 4: a hit on a suspicious path must ban the upstream client (XFF[0]), not the proxy."""
        app = _build_app()
        client = TestClient(app)

        resp = client.get(
            "/.env.bak",
            headers={"X-Forwarded-For": "8.8.8.8, 10.11.0.17"},
        )
        assert resp.status_code == 403

        middleware = _get_middleware(client)
        assert "8.8.8.8" in middleware.blocked_ips
        # The proxy must remain free so legitimate traffic from the same proxy still flows.
        assert "10.11.0.17" not in middleware.blocked_ips
        assert "testclient" not in middleware.blocked_ips


class TestFallbackWhenNoForwardedHeader:
    def test_falls_back_to_request_client_host(self):
        """RED 5: without X-Forwarded-For, the middleware uses request.client.host."""
        app = _build_app()
        client = TestClient(app)

        # No XFF, no X-Real-IP. Suspicious path triggers a block on the direct client.
        resp = client.get("/.env.bak")
        assert resp.status_code == 403

        middleware = _get_middleware(client)
        # TestClient's default host string is "testclient" — that's the only signal we have.
        assert "testclient" in middleware.blocked_ips

    def test_x_real_ip_used_when_x_forwarded_for_missing(self):
        """Bonus coverage: X-Real-IP is honored when X-Forwarded-For is absent."""
        app = _build_app()
        client = TestClient(app)

        resp = client.get(
            "/.env.bak",
            headers={"X-Real-IP": "5.5.5.5"},
        )
        assert resp.status_code == 403

        middleware = _get_middleware(client)
        assert "5.5.5.5" in middleware.blocked_ips
