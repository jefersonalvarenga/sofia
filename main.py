"""
Sofia — EasyScale Patient Conversational Agent
POST /v1/sofia  — main conversational endpoint
GET  /v1/health — health check
v2.0.0
"""

import time
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings, init_dspy
from app.core.security import SecurityMiddleware, AccessLogMiddleware
from app.core.telemetry import log
from app.session.models import SofiaRequest, SofiaResponse

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
        from app.graph.sofia_graph import sofia_graph  # noqa: F401
        settings = get_settings()
        log.info("sofia.ready", version=settings.sofia_version)
    except Exception as e:
        log.error("sofia.startup.error", error=str(e))

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/v1/health")
async def health():
    settings = get_settings()
    return {
        "status": "online",
        "version": settings.sofia_version,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/v1/sofia", response_model=SofiaResponse)
async def sofia_endpoint(request: SofiaRequest):
    """
    Main Sofia endpoint v2.

    Receives a patient message from n8n, runs the Sofia LangGraph,
    and returns agent_runs — a list of agent executions each with
    messages (to send), data (for n8n actions), and observability fields.

    Required header: X-API-Key
    """
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    log.info("sofia.request", trace_id=trace_id, clinic_id=request.clinic_id,
             remote_jid=request.remote_jid, conversation_type=request.conversation_type)

    from app.graph.sofia_graph import sofia_graph

    try:
        result = sofia_graph.invoke({
            # ---- Inputs from n8n ----
            "instance_id":       request.instance_id,
            "clinic_id":         request.clinic_id,
            "remote_jid":        request.remote_jid,
            "push_name":         request.push_name,
            "message":           request.message,
            "message_type":      request.message_type,
            "wamid":             request.wamid,
            "available_slots":   request.available_slots,
            "conversation_type": request.conversation_type,
            "attribution_id":    request.attribution_id,
            # ---- Observability ----
            "trace_id":          trace_id,
            "language":          "pt-BR",   # overwritten by Router
            # ---- Session context (populated by load_context) ----
            "session_id":        "",
            "clinic_name":       "",
            "assistant_name":    "Sofia",
            "history":           [],
            "conversation_stage": "new",
            "patient_name":      request.push_name,
            "customer_id":       None,
            # ---- Routing (populated by detect_intents) ----
            "detected_intents":  [],
            # ---- Agent outputs (populated by execute_agents) ----
            "agent_runs":        [],
            "requires_human":    False,
        })

        processing_time = (time.time() - start_time) * 1000

        agent_runs = result.get("agent_runs", [])
        requires_human = result.get("requires_human", False)
        conversation_stage = "active"
        language = result.get("language", "pt-BR")

        # Derive conversation_stage from last agent_run
        if agent_runs:
            last_stage = agent_runs[-1].get("conversation_stage")
            if last_stage:
                conversation_stage = last_stage

        log.info("sofia.response", trace_id=trace_id,
                 agents=[r.get("agent") for r in agent_runs],
                 conversation_stage=conversation_stage,
                 processing_ms=round(processing_time, 2))

        return SofiaResponse(
            agent_runs=agent_runs,
            session_id=result.get("session_id", ""),
            conversation_stage=conversation_stage,
            requires_human=requires_human,
            processing_time_ms=processing_time,
            trace_id=trace_id,
            language=language,
        )

    except Exception as e:
        log.error("sofia.error", trace_id=trace_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Sofia Error: {str(e)}")
