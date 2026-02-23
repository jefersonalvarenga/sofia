"""
Sofia — EasyScale Patient Conversational Agent
POST /v1/sofia  — main conversational endpoint
GET  /v1/health — health check
"""

import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings, init_dspy
from app.core.security import SecurityMiddleware, AccessLogMiddleware
from app.session.models import SofiaRequest, SofiaResponse

# ============================================================================
# App initialization
# ============================================================================

app = FastAPI(
    title="Sofia — EasyScale Patient Agent",
    description="Agente conversacional de atendimento de pacientes via WhatsApp",
    version="1.0.0",
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
    print("Sofia starting...")
    try:
        init_dspy()
        # Import graph here to trigger agent singleton initialization after DSPy is ready
        from app.graph.sofia_graph import sofia_graph  # noqa: F401
        settings = get_settings()
        print(f"Sofia {settings.sofia_version} ready.")
    except Exception as e:
        print(f"Startup error: {e}")

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
    Main Sofia endpoint.

    Receives a patient message from n8n, runs the Sofia LangGraph,
    and returns the agent response.

    Required header: X-API-Key
    """
    start_time = time.time()

    # Lazy import to ensure DSPy is initialized before graph import
    from app.graph.sofia_graph import sofia_graph

    try:
        result = sofia_graph.invoke({
            "instance_id": request.instance_id,
            "clinic_id": request.clinic_id,
            "remote_jid": request.remote_jid,
            "push_name": request.push_name,
            "message": request.message,
            "message_type": request.message_type,
            "wamid": request.wamid,
            "available_slots": request.available_slots,
            # State fields initialized to safe defaults
            "session_id": "",
            "clinic_name": "",
            "assistant_name": "Sofia",
            "services_context": "{}",
            "business_rules": "[]",
            "history": [],
            "conversation_stage": "new",
            "patient_name": request.push_name,
            "customer_id": None,
            "intent": None,
            "response_message": None,
            "agent_name": None,
            "requires_human": False,
            "appointment_created": None,
            "reasoning": None,
            "processing_time_ms": 0.0,
        })

        processing_time = (time.time() - start_time) * 1000

        response_message = result.get("response_message") or "Como posso ajudar?"
        agent_name = result.get("agent_name") or "FAQResponder"
        should_send = bool(response_message)

        return SofiaResponse(
            response_message=response_message,
            should_send=should_send,
            agent_name=agent_name,
            conversation_stage=result.get("conversation_stage") or "active",
            session_id=result.get("session_id") or "",
            requires_human=result.get("requires_human", False),
            appointment_created=result.get("appointment_created"),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sofia Error: {str(e)}")
