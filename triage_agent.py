# triage_agent.py — SOLO classifier (niente routing)
import os, re
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ---- Config ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ---- LLM deterministico per classificazione ----
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0.0)

# ---- FastAPI ----
app = FastAPI(title="WeSafe Triage (Classifier Only)")

# ---- Schemi ----
class ChatIn(BaseModel):
    session_id: str
    message: str

class ClassifyIn(BaseModel):
    text: str

class ClassifyOut(BaseModel):
    klass: int  # 1 | 2

# ---- Prompt di classificazione (rigido) ----
SYSTEM_PROMPT = (
    "Sei l'Assistente WeSafe per la certificazione notarile e l’analisi di visure. "
    "Parla SEMPRE in italiano, con tono professionale e sintetico.\n\n"
    "Classifica l'input dell'utente in:\n"
    "  1. L'utente non sa quali documenti servono, chiede informazioni generiche su documenti.\n"
    "  2. L'utente sa già quali documenti servono e vuole un documento specifico.\n\n"
    "RISPOSTA OBBLIGATORIA:\n"
    "Scrivi SOLO <K>1</K> oppure <K>2</K>. In dubbio, <K>1</K>."
)

def _classify_raw(text: str) -> int:
    print("CLASSIFY RAW")
    out = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)])
    raw = (getattr(out, "content", "") or "").strip()
    m = re.search(r"<K>\s*([12])\s*</K>", raw) or re.search(r"\b([12])\b", raw)
    if m:
        return int(m.group(1))
    # fallback euristico minimale
    t = text.lower()
    return 2 if any(k in t for k in [
        "visura","catastale","relazione preliminare","certificazione notarile","art. 567",
        "trascrizione","iscrizione","atto notarile","provenienza","ipoteca","mutuo","gravami"
    ]) else 1

# ---- Endpoints ----
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Stub per compat UI (la tua UI chiama /test_retriever sull’API base; qui rispondiamo tranquilli)
@app.get("/test_retriever")
def test_retriever(q: str = ""):
    print("TEST RETRIEVER")
    return {"docs": []}

# Preface per messaggio vuoto
@app.post("/chat")
def chat_preface(body: ChatIn):
    print("CHAT PREFACE")
    msg = (body.message or "").strip()
    if not msg:
        return {
            "reply": "Ciao! 👋 Sono un agente di WeSafe!\nSai già quali documenti ti servono o hai bisogno di aiuto?"
        }
    # Se qualcuno manda testo a /chat, rispondi minimo e ricorda che il router userà /classify
    k = _classify_raw(msg)
   
    return {"reply": f"(triage) Classe stimata: <K>{k}</K>. Il router userà /classify per instradarti."}

@app.post("/classify", response_model=ClassifyOut)
def classify(payload: ClassifyIn):
    text = (payload.text or "").strip()

    if not text:
        k = 1
    else:
        k = _classify_raw(text)

    print(f"[TRIAGE] CLASSE DECISA = {k} (input='{text[:80]}...')")

    return ClassifyOut(klass=k)

