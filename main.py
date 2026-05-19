"""
Sofia — EasyScale Patient Conversational Agent
POST /v1/sofia  — main conversational endpoint
GET  /v1/health — health check
v2.0.0
"""

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings, init_dspy
from app.core.security import SecurityMiddleware, AccessLogMiddleware
from app.core.telemetry import log
from app.iris.webhook import router as iris_router

# ============================================================================
# App initialization
# ============================================================================

app = FastAPI(
    title="Sofia — EasyScale Patient Agent",
    description="Agente conversacional de atendimento de pacientes via WhatsApp",
    version="2.0.0",
)

app.add_middleware(SecurityMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    log.info("sofia.startup", msg="Sofia starting...")
    try:
        init_dspy()
        from app.iris.pipeline import iris_graph  # noqa: F401
        settings = get_settings()
        log.info("sofia.ready", version=settings.sofia_version)
    except Exception as e:
        log.error("sofia.startup.error", error=str(e))

# ============================================================================
# Endpoints
# ============================================================================

app.include_router(iris_router)


@app.get("/v1/health")
async def health():
    settings = get_settings()
    return {
        "status": "online",
        "version": settings.sofia_version,
        "timestamp": datetime.utcnow().isoformat(),
    }


# /v1/sofia endpoint removed — Iris pipeline at /iris/* is canonical.
# Legacy sofia_graph deleted in feat/pipeline-integration.
