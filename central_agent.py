# central_agent.py – router/hub finale (LangGraph + VectorStoreRetrieverMemory)

import os
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel

# LangChain / LangGraph
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langgraph.graph import StateGraph, END

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
# Memoria: VectorStoreRetrieverMemory su Qdrant
# -------------------------
# In questa versione base, la memoria NON separa per sessione: salviamo
# i messaggi prefissando il SID nel testo, così il retriever potrà comunque
# sfruttare il contenuto (opzione semplice senza metadata).
emb = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://ollama:11434")
qdrant_client = QdrantClient(url="http://qdrant:6333")
vectordb = Qdrant(client=qdrant_client, collection_name="chat_memory", embeddings=emb)
memory = VectorStoreRetrieverMemory(retriever=vectordb.as_retriever())

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

def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Salva input/output nella memoria vettoriale.
    NOTA: qui non distinguiamo per sessione con metadati;
    prefissiamo il SID nel testo d'input per dare un minimo di separazione logica.
    """
    sid = state["session_id"]
    # Salviamo sempre l'ultimo input; l'output viene salvato se presente.
    memory.save_context(
        {"input": f"[{sid}] {state['input']}"},
        {"output": state.get("reply", "")}
    )
    return state

async def base_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo che chiama l'agente RAG 'base' e ritorna la risposta.
    """
    sid = state["session_id"]
    user_msg = state["input"]
    reply = await _call_base_agent(sid, user_msg)
    return {"reply": reply}

def special_agent_node(_: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder per l'agente 'special' (non ancora implementato).
    """
    return {"reply": "⚠️ Questo agente (special) non è stato ancora sviluppato."}

# -------------------------
# Costruzione del grafo
# -------------------------
workflow = StateGraph()
workflow.add_node("triage", triage_node)
workflow.add_node("memory", memory_node)
workflow.add_node("base", base_agent_node)
workflow.add_node("special", special_agent_node)

# triage → memory (sempre), poi rotta condizionale in base alla klass
workflow.add_edge("triage", "memory")
workflow.add_conditional_edges(
    "memory",
    lambda state: "special" if state.get("klass") == "2" and SPECIAL_AGENT_URL else "base"
)
workflow.add_edge("base", END)
workflow.add_edge("special", END)

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
    return ChatOut(reply=str(result.get("reply") or "(nessuna risposta)"))

@app.post("/reset_session")
def reset_session(session_id: str = Query(...)):
    """
    Con VectorStoreRetrieverMemory non c'è una sessione in RAM da svuotare.
    Se vuoi davvero "pulire", dovresti salvare i messaggi con metadati (sid)
    e poi cancellare i punti relativi a quel sid da Qdrant.
    """
    return {"ok": True, "note": "reset_session non implementa la cancellazione su Qdrant in questa versione."}

@app.get("/session/{sid}/history")
def get_history(sid: str):
    """
    Non implementato: VectorStoreRetrieverMemory non espone una history lineare per sessione.
    Per avere una cronologia per SID, salva i messaggi in Qdrant con metadata={'sid': ...}
    e fai query/filtri su quel campo.
    """
    return {"history": "non implementato con VectorStoreRetrieverMemory (vedi nota in codice)"}
