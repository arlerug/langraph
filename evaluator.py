from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import ChatMessageHistory
import os, re

# ---------------- Config ----------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

SYSTEM_PROMPT = (
    "Sei l'Assistente WeSafe per la certificazione notarile e l’analisi di visure. "
    "Parla SEMPRE in italiano, con tono professionale e sintetico.\n\n"
    "Il tuo compito è classificare l'input dell'utente in una delle due categorie:\n\n"
    "  1. **Richiesta autonoma** → L'utente formula una domanda o richiesta chiara e completa, "
    "che può essere compresa senza bisogno di leggere i messaggi precedenti. "
    "Esempi: 'Cos'è una visura catastale?', 'Voglio sapere cosa contiene un atto notarile'.\n\n"
    "  2. **Messaggio contestuale** → L'utente scrive qualcosa che ha senso SOLO se consideri la conversazione precedente. "
    "Sono frasi brevi, vaghe, o chiarificazioni legate a un turno prima. "
    "Esempi: 'in che senso?', 'cosa intendi?', 'puoi spiegare meglio?', 'quello che hai detto prima', 'e per l'altra visura?', 'riassumi', 'dillo in una frase'.\n\n"
    "RISPOSTA OBBLIGATORIA:\n"
    "Scrivi SOLO <K>1</K> oppure <K>2</K>."
)

# ---------------- FastAPI ----------------
app = FastAPI(title="WeSafe Evaluator con Memoria")

# ---------------- Schemi ----------------
class ChatIn(BaseModel):
    session_id: str = "default"
    message: str
    history: str|None = None
    reply: str|None = None  

class ChatOut(BaseModel):
    eval_class: int  # 1 | 2



# ---------------- Funzioni ----------------
def _evaluate(text: str, history: str | None, agent_reply: str | None) -> int:
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.0,
        num_ctx=NUM_CTX,
    )
    print("[EVALUATOR] history:", history)
    history_text = history 
    print("[EVALUATOR] history_text:", history_text)
    # Prompt con memoria
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Conversazione finora:\n{history_text}\n\n"
        f"Nuovo input utente:\n{text}\n\nRisposta:"
    )

    print(f"[EVAL] prompt:\n{prompt}")

    reply = llm.invoke(prompt)
    raw = str(reply).strip()
    print(f"[EVAL] RAW LLM OUTPUT: {raw}")

    if "2" in raw:
        eval_class = 2
    else:
        eval_class = 1


    return eval_class

# ---------------- Endpoint ----------------
@app.post("/evaluate", response_model=ChatOut)
def evaluate(in_data: ChatIn) -> ChatOut:
    try:
        print("[EVALUATOR] IN_DATA:", in_data.message, in_data.history, in_data.reply)
        eval_class = _evaluate(in_data.message, in_data.history, in_data.reply)
        print(f"[EVAL] session={in_data.session_id} eval_class={eval_class}")
        return ChatOut(eval_class=eval_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella valutazione: {e}")

