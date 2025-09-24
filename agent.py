# agent.py
import os
import threading
import traceback
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LLM base (non-chat) di Ollama
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import OllamaEmbeddings

from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

# ---------------- Config (env) ----------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # modello base, non -instruct
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_legale_it")

EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

SYSTEM_PROMPT = """
Sei l'Assistente WeSafe per la certificazione notarile e l’analisi di visure.
Parla SEMPRE in italiano, con tono professionale, sintetico e orientato all’azione.

---

🎯 Competenze:
- Certificazione notarile: documento completo con almeno 20 anni di storia dell’immobile (proprietà, atti di provenienza, gravami). Obbligatoria nelle procedure esecutive.
- Copia di un atto: riproduzione di un singolo atto notarile o giudiziario (compravendita, donazione, mutuo). Ha valore di prova legale puntuale.
- Ipotecario per immobile: elenca ipoteche, pignoramenti e altre formalità su uno specifico immobile.
- Ipotecario per soggetto: elenca tutte le formalità registrate a carico di una persona o società.
- Mappa catastale: estratto grafico che mostra particelle, confini e posizione degli immobili.
- Nota di iscrizione: atto per iscrivere una formalità (es. ipoteca) in Conservatoria.
- Nota di trascrizione compravendita: atto che certifica il passaggio di proprietà a seguito di una compravendita.
- Visura catastale: descrive i dati identificativi e storici di un immobile (fabbricati o terreni), intestatari e variazioni catastali.
- Visura ipocatastale attuale: unisce dati catastali e ipotecari per la “fotografia” attuale di un immobile o soggetto.

---

🎯 Obiettivi:
1. Capire il bisogno espresso dall’utente.
2. Indicare i documenti più utili per soddisfarlo, spiegando brevemente:
   - a cosa servono,
   - quali informazioni contengono,
   - in quali casi vengono richiesti.
3. Rispondere in modo chiaro e contestualizzato alle domande sui documenti.

---

⚖️ Regole:
- Non chiedere mai dati personali (codice fiscale, data di nascita, indirizzi).
- Non inventare documenti o procedure: usa SOLO quelli disponibili nell’elenco.
- Se l’esigenza non è chiara, chiedi chiarimenti ma proponi comunque un documento iniziale utile.
- Chiudi sempre con la sezione “📂 Documenti consigliati:” seguita dai documenti pertinenti, e chiedi conferma all’utente.

---

📝 Stile:
- Risposte brevi, professionali, focalizzate all’azione.
- Linguaggio semplice, ma autorevole.

---

⚙️ Operatività:
- Usa SOLO il contesto recuperato. Se insufficiente, dillo chiaramente e proponi un documento iniziale utile.
"""


# ---------------- App & stato ----------------
app = FastAPI(title="Minimal Agent", version="0.1")
_status = "initializing"
_last_error_tb = None
_components = None
_sessions: Dict[str, ChatMessageHistory] = {}

# ---------------- Utils ----------------
def _history(session_id: str) -> ChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = ChatMessageHistory()
    return _sessions[session_id]

def _init_components():
    """Inizializza LLM + retriever Qdrant (OllamaEmbeddings)."""
    global _components, _status, _last_error_tb
    if _components is not None:
        return _components
    try:
        # 1) LLM (Ollama modello base)
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_ctx=NUM_CTX,
        )

        # 2) Qdrant client
        qdrant_client = QdrantClient(url=QDRANT_URL)

        # Auto-detect chiave testo nel payload
        candidate_keys = ["page_content", "text", "content", "chunk", "body", "document", "raw_text"]
        detected_key = None
        try:
            points, _ = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                pay = points[0].payload or {}
                for k in candidate_keys:
                    v = pay.get(k)
                    if isinstance(v, str) and v.strip():
                        detected_key = k
                        break
        except Exception:
            detected_key = None

        # 3) Embeddings via Ollama (mxbai-embed-large)
        emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

        # Verifica dimensione contro la collection
        probe_dim = len(emb.embed_query("probe"))
        try:
            cinfo = qdrant_client.get_collection(QDRANT_COLLECTION)
            vec_cfg = getattr(cinfo.config.params, "vectors", None)
            qdrant_dim = getattr(vec_cfg, "size", None) if vec_cfg else None
        except Exception:
            qdrant_dim = None

        if qdrant_dim is not None and qdrant_dim != probe_dim:
            _status = f"error: embedding dim mismatch (collection={qdrant_dim}, model={probe_dim})"
            _last_error_tb = None
            _components = None
            return None

        vectordb = Qdrant(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embeddings=emb,
            content_payload_key=detected_key or "page_content",
            metadata_payload_key="metadata",
        )
        retriever = vectordb.as_retriever(
            search_kwargs={
                "k": 8,
                # "search_type": "mmr", "lambda_mult": 0.5,  # opzionale
                # "score_threshold": 0.2,                    # opzionale
            }
        )

        _components = {
            "llm": llm,
            "retriever": retriever,
            "qdrant_client": qdrant_client,
            "collection_name": QDRANT_COLLECTION,
            "content_key": detected_key or "page_content",
        }
        _status = "ready"
        _last_error_tb = None
        return _components

    except Exception as e:
        _status = f"error: {e}"
        _last_error_tb = traceback.format_exc()
        return None

def _warmup_bg():
    _init_components()

# ---------------- Schemi ----------------
class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class ChatResponse(BaseModel):
    reply: str

# ---------------- Eventi FastAPI ----------------
@app.on_event("startup")
def _startup():
    global _status
    _status = "initializing"
    threading.Thread(target=_warmup_bg, daemon=True).start()

# ---------------- Endpoint ----------------
@app.get("/healthz")
def healthz():
    return {"status": _status}

@app.get("/debugz")
def debugz():
    return {"status": _status, "traceback": _last_error_tb}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    print("APP.POST \chat")
    print(f"[CHAT] session_id={req.session_id} message={req.message!r}")

    comps = _init_components()
    if not comps:
        print("COMPONENTI NON INIZIALIZZATE")
        print(f"[INIT] componenti non inizializzati: status={_status}")
        raise HTTPException(status_code=503, detail=f"Agent non inizializzato: {_status}")

    # Recupero passaggi rilevanti
    try:
        print("RECUPERO DOCUMENTI")
        docs = comps["retriever"].get_relevant_documents(req.message)
        print(f"[RETRIEVER] trovati {len(docs)} documenti")
        for i, d in enumerate(docs[:3]):
            print(f"[DOC {i}] {d.page_content[:200]}...")
        context = "\n\n".join([d.page_content[:1300] for d in docs])
    except Exception as e:
        print(f"[RETRIEVER] errore: {e}")
        context = ""

    # Prompt per il modello
    user_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Contesto (estratti da knowledge base):\n{context}\n\n"
        f"Utente: {req.message}\nAssistente:"
    )
    print(f"[PROMPT preview] {user_text[:300]}...") 
    print("USATO AGENT-BASE") # solo i primi 300 caratteri

    # Invocazione LLM
    try:
        reply = comps["llm"].invoke(user_text)
        print(f"[LLM] reply ricevuta, tipo={type(reply)}, len={len(str(reply))}")
    except Exception as e:
        print(f"[LLM] errore durante l'invocazione: {e}")
        raise HTTPException(status_code=500, detail=f"Errore LLM: {e}")

    return ChatResponse(reply=str(reply))

@app.get("/qdrantz")
def qdrantz():
    print("QDRANTZ")
    comps = _init_components()
    qc = comps.get("qdrant_client")
    cname = comps.get("collection_name")
    if not qc or not cname:
        return {"ok": False, "error": "qdrant non inizializzato"}
    try:
        info = qc.get_collection(cname)
        cnt = qc.count(cname, exact=True).count
        return {
            "ok": True,
            "url": QDRANT_URL,
            "collection": cname,
            "points": cnt,
            "status": str(info.status),
            "vectors": getattr(info.config.params, "vectors", None).__dict__ if hasattr(info.config.params, "vectors") else None,
        }
    except Exception as e:
        return {"ok": False, "collection": cname, "error": str(e)}

@app.get("/test_retriever")
def test_retriever(q: str = "visura catastale"):
    comps = _init_components()
    retriever = comps.get("retriever")
    if not retriever:
        return {"ok": False, "error": "retriever non disponibile"}
    try:
        docs = retriever.get_relevant_documents(q)
        return {"ok": True, "query": q, "num_docs": len(docs), "docs": [d.page_content[:200] for d in docs]}
    except Exception as e:
        return {"ok": False, "query": q, "error": str(e)}

@app.get("/qpeek")
def qpeek(n: int = 3):
    comps = _init_components()
    qc = comps.get("qdrant_client")
    cname = comps.get("collection_name")
    print("QPEEK")
    if not qc or not cname:
        return {"ok": False, "error": "qdrant non inizializzato"}
    try:
        points, _ = qc.scroll(
            collection_name=cname,
            limit=n,
            with_payload=True,
            with_vectors=False,
        )
        samples = []
        for p in points:
            pay = p.payload or {}
            text_like = None
            for k in ["page_content", "text", "content", "body", "document", "chunk", "raw_text"]:
                v = pay.get(k)
                if isinstance(v, str) and v.strip():
                    text_like = {"key": k, "preview": v[:200]}
                    break
            samples.append({"id": p.id, "keys": list(pay.keys()), "text_like": text_like})
        return {"ok": True, "collection": cname, "samples": samples}
    except Exception as e:
        return {"ok": False, "error": str(e)}
