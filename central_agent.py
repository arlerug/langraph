# central_agent.py – router/hub finale (LangGraph + VectorStoreRetrieverMemory)

import os
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel

# LangChain / LangGraph
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional


# -------------------------
# Config da env
# -------------------------
DEBUG_TRIAGE = os.getenv("DEBUG_TRIAGE", "0") == "1"
TRIAGE_URL = os.getenv("TRIAGE_URL", "http://triage:8000")          # senza /chat
BASE_AGENT_URL = os.getenv("BASE_AGENT_URL", "http://agent_base:8000")
SPECIAL_AGENT_URL = os.getenv("SPECIAL_AGENT_URL", "")               # opzionale
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "300"))

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="WeSafe Central Router", version="1.0")

class ChatIn(BaseModel):
    session_id: str
    message: str

class ChatOut(BaseModel):
    reply: str

# -------------------------
# Memoria: short-term in RAM per sessione
# -------------------------
_sessions: dict[str, ChatMessageHistory] = {}

def _get_history(sid: str) -> ChatMessageHistory:
    if sid not in _sessions:
        _sessions[sid] = ChatMessageHistory()
    return _sessions[sid]

# -------------------------
# Helpers HTTP
# -------------------------
async def _triage_classify(text: str) -> str:
    """Chiama il Triage Agent per ottenere 'klass' = '1' | '2'."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        r = await client.post(f"{TRIAGE_URL.rstrip('/')}/classify", json={"text": text})
        r.raise_for_status()
        k = str((r.json() or {}).get("klass") or "1")
        return "1" if k not in ("1", "2") else k

async def _call_base_agent(session_id: str, message: str) -> str:
    """Chiama l'agente base (RAG) e ritorna la reply come stringa."""
    payload = {"session_id": session_id, "message": message}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        r = await client.post(f"{BASE_AGENT_URL.rstrip('/')}/chat", json=payload)
        r.raise_for_status()
        out = r.json() if "application/json" in (r.headers.get("content-type") or "") else {"reply": r.text}
    return str(out.get("reply") or out.get("error") or "")

# -------------------------
# Nodi del grafo (LangGraph)
# -------------------------
async def triage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide la classe della richiesta:
      '1' → informativa / discovery → agente base
      '2' → richiesta documento specifico → agente special (placeholder)
    """
    text = state.get("input", "")
    klass = await _triage_classify(text)
    return {"klass": klass}

async def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sid = state["session_id"]
    history = _get_history(sid)
    history.add_user_message(state["input"])
    if "reply" in state:
        history.add_ai_message(state["reply"])
    print(f"[MEMORY] updated short-term memory for session {sid}")
    return {}


async def base_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo che chiama l'agente RAG 'base' e ritorna la risposta,
    passando anche la memoria condivisa (short-term centralizzata).
    """
    sid = state["session_id"]
    user_msg = state["input"]

    # recupera la history da central
    history = _get_history(sid)
    past_msgs = [f"{m.type.capitalize()}: {m.content}" for m in history.messages[-5:]]  
    history_text = "\n".join(past_msgs)
    print(f"[CENTRAL] past_msgs for session def base_agent:", history_text)

    # payload completo
    payload = {
        "session_id": sid,
        "message": user_msg,
        "history": history_text  # nuovo campo
    }
    print(f"[CENTRAL] payload for base_agent:", payload)

    # invoca l’agente
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        r = await client.post(f"{BASE_AGENT_URL.rstrip('/')}/chat", json=payload)
        r.raise_for_status()
        out = r.json() if "application/json" in (r.headers.get("content-type") or "") else {"reply": r.text}

    reply = str(out.get("reply") or out.get("error") or "")
    return {"reply": reply}


async def special_agent_node(_: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder per l'agente 'special' (non ancora implementato).
    """
    return {"reply": "⚠️ Questo agente (special) non è stato ancora sviluppato."}

# -------------------------
# Costruzione del grafo
# -------------------------
class State(TypedDict, total=False):
    session_id: str
    input: str
    reply: Optional[str]
    klass: Optional[str]

workflow = StateGraph(State)
workflow.add_node("triage", triage_node)
workflow.add_node("memory", memory_node)
workflow.add_node("base", base_agent_node)
workflow.add_node("special", special_agent_node)

# triage decide il percorso
workflow.add_conditional_edges(
    "triage",
    lambda state: "special" if state.get("klass") == "2" and SPECIAL_AGENT_URL else "base"
)

# dopo che gli agenti producono reply → salva in memoria
workflow.add_edge("base", "memory")
workflow.add_edge("special", "memory")

# memory → fine
workflow.add_edge("memory", END)

# entrypoint
workflow.add_edge(START, "triage")


app_graph = workflow.compile()

# -------------------------
# Endpoint HTTP
# -------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    """
    Entry-point: passa l'input al grafo LangGraph.
    """
    msg = (body.message or "").strip()
    if not msg:
        # Preface "neutro" se vuoi mantenere il comportamento di triage/chat a messaggio vuoto.
        return ChatOut(reply="Ciao! 👋 Sai già quali documenti ti servono o hai bisogno di aiuto?")

    result = await app_graph.ainvoke({
        "session_id": body.session_id,
        "input": msg
    })
    print(f"[CENTRAL] RESULT: {result} ")
    return ChatOut(reply=str(result.get("reply") or "(nessuna risposta)"))

@app.post("/reset_session")
def reset_session(session_id: str = Query(...)):
    _sessions.pop(session_id, None)
    return {"ok": True, "cleared": session_id}

@app.get("/session/{sid}/history")
def get_history(sid: str):
    history = _get_history(sid)
    msgs = [f"{m.type}: {m.content}" for m in history.messages]
    return {"history": msgs}
