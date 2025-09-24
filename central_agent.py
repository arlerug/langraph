# central_agent.py – router/hub finale
import os
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DEBUG_TRIAGE = os.getenv("DEBUG_TRIAGE", "0") == "1"
TRIAGE_URL = os.getenv("TRIAGE_URL", "http://triage:8000")          # senza /chat
BASE_AGENT_URL = os.getenv("BASE_AGENT_URL", "http://agent_base:8000")
SPECIAL_AGENT_URL = os.getenv("SPECIAL_AGENT_URL", "")               # opzionale
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "300"))
MAX_HOPS = int(os.getenv("MAX_HOPS", "4"))

app = FastAPI(title="WeSafe Central Router", version="1.0")

class ChatIn(BaseModel):
    session_id: str
    message: str

class ChatOut(BaseModel):
    reply: str

# Stato in-memory: per demo
SESSIONS: Dict[str, Dict[str, Any]] = {}  # sid -> {klass, current_agent, meta}

def _get_sess(sid: str) -> Dict[str, Any]:
    if sid not in SESSIONS:
        SESSIONS[sid] = {"klass": None, "current_agent": None, "meta": {}}
        print(f"NEW SESSION: {sid}")
    return SESSIONS[sid]

async def _triage_classify(text: str) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        print("CALL TRIAGE CLASSIFY")
        r = await client.post(f"{TRIAGE_URL.rstrip('/')}/classify", json={"text": text})
        r.raise_for_status()
        k = str((r.json() or {}).get("klass") or "1")
        return "1" if k not in ("1", "2") else k

async def _post_agent(url_base: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        print(f"CALL AGENT: {url_base} with payload keys {list(payload.keys())}")
        r = await client.post(f"{url_base.rstrip('/')}/chat", json=payload)
        r.raise_for_status()
        # se non è JSON valido, solleva e verrà gestito dal caller
        return r.json() if "application/json" in (r.headers.get("content-type") or "") else {"reply": r.text}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "sessions": len(SESSIONS)}

@app.get("/session/{sid}")
def session_state(sid: str):
    return _get_sess(sid)

@app.post("/reset_session")
def reset_session(session_id: str = Query(...)):
    print(f"RESET SESSION: {session_id}")
    SESSIONS.pop(session_id, None)
    return {"ok": True}

# proxy per la UI: /test_retriever va all'agente base
@app.get("/test_retriever")
async def proxy_retriever(q: str = ""):
    print("TEST RETRIEVER")
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            r = await client.get(f"{BASE_AGENT_URL.rstrip('/')}/test_retriever", params={"q": q})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"ok": False, "error": f"proxy_error: {e}"}

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    s = _get_sess(body.session_id)
    msg = (body.message or "").strip()

    # 1) Messaggio vuoto → preface dal triage (senza innescare loop)
    if not msg:
        print("EMPTY MESSAGE: PREFACE FROM TRIAGE")
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
                r = await client.post(
                    f"{TRIAGE_URL.rstrip('/')}/chat",
                    json={"session_id": body.session_id, "message": ""}
                )
                r.raise_for_status()
                j = r.json()
                return ChatOut(reply=str(j.get("reply") or "Ciao! 👋"))
        except Exception:
            return ChatOut(reply="Ciao! 👋 Sai già quali documenti ti servono o hai bisogno di aiuto?")

    # 2) Classificazione al primo messaggio non-vuoto
    if not s.get("klass"):
        print("FIRST NON-EMPTY MESSAGE: CLASSIFY VIA TRIAGE")
        try:
            s["klass"] = await _triage_classify(msg)  # "1" | "2"
        except Exception:
            s["klass"] = "1"  # fallback prudente

    # Prefix di debug (mostrato solo una volta alla prima risposta utile)
    prefix = f"(debug: klass={s['klass']})\n\n" if DEBUG_TRIAGE else ""
    prefix_applied = False

    # 3) Scegli agente iniziale se non impostato
    if not s.get("current_agent"):
        print("VALORE DI KLASS E AGENTE INIZIALE")
        print(s["klass"])
        print(SPECIAL_AGENT_URL)
        if s["klass"] == "2" and SPECIAL_AGENT_URL:
            s["current_agent"] = "special"
        else:
            s["current_agent"] = "base"
    print("NON HO CAPITO CHE SUCCEDE QUI" + f"[SESSION {body.session_id}] HOP 0: klass={s['klass']}, current_agent={s['current_agent']}")

    # 4) Orchestrazione a hop limitati (handoff tra agenti)
    last_reply = ""
    hops = 0
    while hops < MAX_HOPS:
        print("RIMBALZO")
        hops += 1
        agent_key = s["current_agent"]
        url = SPECIAL_AGENT_URL if agent_key == "special" else BASE_AGENT_URL
        payload = {"session_id": body.session_id, "message": msg}

        try:
            out = await _post_agent(url, payload)
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"errore chiamando {agent_key}: HTTP {e.response.status_code}")
        except Exception as e:
            raise HTTPException(502, f"errore chiamando {agent_key}: {e}")

        # Normalizzazione output
        reply  = str(out.get("reply") or out.get("error") or "")
        done   = bool(out.get("done", True))     # default True per evitare loop
        handoff = out.get("handoff")             # "base" | "special" | None
        ctxu   = out.get("context_updates") or {}

        # Applica il prefix di debug alla PRIMA risposta non vuota
        if DEBUG_TRIAGE and not prefix_applied and reply:
            reply = prefix + reply
            prefix_applied = True

        last_reply = reply or last_reply

        # Aggiorna stato sessione
        if isinstance(ctxu.get("klass"), str):
            s["klass"] = ctxu["klass"]
        if isinstance(ctxu.get("meta"), dict):
            s["meta"].update(ctxu["meta"])

        # Handoff esplicito
        if handoff in ("base", "special"):
            s["current_agent"] = handoff
            continue

        # (Opzionale) riallinea l'agente se la klass è cambiata senza handoff
        if handoff is None and s.get("klass") in ("1", "2"):
            desired = "special" if (s["klass"] == "2" and SPECIAL_AGENT_URL) else "base"
            if desired != s.get("current_agent"):
                s["current_agent"] = desired
                continue  # ripeti immediatamente con l'agente corretto

        # Se l'agente ha finito (o default), chiudi
        if done:
            return ChatOut(reply=last_reply or "(nessuna risposta)")

        # Nessun handoff e non done → esci per evitare loop
        break

    # 5) Ritorno finale: se non è mai stato possibile applicare il prefix, prova ora
    if DEBUG_TRIAGE and not prefix_applied and last_reply:
        last_reply = prefix + last_reply
    return ChatOut(reply=last_reply or "(nessuna risposta)")

