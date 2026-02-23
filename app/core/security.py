"""
Security Middleware for Sofia API
"""

import time
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def _get_api_key() -> Optional[str]:
    """Lazy import to avoid circular dependency at module load time."""
    from app.core.config import get_settings
    return get_settings().api_key


SUSPICIOUS_PATHS = [
    ".git", ".env", "aws", "terraform", "docker", "wp-admin", "wp-content",
    "phpinfo", "credentials", ".aws", "root/", "admin", ".ssh",
    "backup", "database", ".sql", ".tar", ".zip", "passwd", "shadow"
]

SUSPICIOUS_EXTENSIONS = [
    ".php", ".asp", ".aspx", ".jsp", ".cgi", ".sh", ".bat", ".cmd"
]

BLOCKED_USER_AGENTS = [
    "nikto", "sqlmap", "nmap", "masscan", "nessus", "openvas",
    "acunetix", "burp", "zaproxy", "metasploit"
]


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > minute_ago
        ]
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        self.requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        return max(0, self.requests_per_minute - len(self.requests[client_ip]))


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit: int = 60):
        super().__init__(app)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit)
        self.blocked_ips: Dict[str, datetime] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        path = request.url.path.lower()
        user_agent = request.headers.get("user-agent", "").lower()

        if client_ip in self.blocked_ips:
            if datetime.now() < self.blocked_ips[client_ip]:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "IP temporarily blocked due to suspicious activity"}
                )
            else:
                del self.blocked_ips[client_ip]

        if any(suspicious in path for suspicious in SUSPICIOUS_PATHS):
            self._block_ip(client_ip, minutes=30)
            return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": "Access denied"})

        if any(path.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS):
            self._block_ip(client_ip, minutes=30)
            return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": "Access denied"})

        if any(blocked in user_agent for blocked in BLOCKED_USER_AGENTS):
            self._block_ip(client_ip, minutes=60)
            return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": "Access denied"})

        if path.startswith("/v1/") and path != "/v1/health":
            required_key = _get_api_key()
            if required_key:
                provided_key = request.headers.get("X-API-Key", "")
                if provided_key != required_key:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or missing API key"}
                    )

        if path.startswith("/v1/"):
            if not self.rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Please try again later.", "retry_after": 60},
                    headers={"Retry-After": "60"}
                )

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        if path.startswith("/v1/"):
            remaining = self.rate_limiter.get_remaining(client_ip)
            response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _block_ip(self, ip: str, minutes: int):
        self.blocked_ips[ip] = datetime.now() + timedelta(minutes=minutes)
        print(f"SECURITY: Blocked IP {ip} for {minutes} min")


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        path = request.url.path
        if path.startswith("/v1/") or response.status_code >= 400:
            print(
                f"{request.method} {path} -> {response.status_code} "
                f"({duration*1000:.0f}ms) [{request.client.host}]"
            )
        return response
