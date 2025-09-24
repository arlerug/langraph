import os
import requests
import streamlit as st

st.set_page_config(page_title="WeSafe – Chat", page_icon="🗂️", layout="centered")

# ---------------- Stato (FIX: inizializza prima di tutto) ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    print("1")
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit"
    print("1")
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
    print("1")
if "preface_shown" not in st.session_state:
    st.session_state.preface_shown = False  # FIX: default esplicito
    print("1")

# ---------------- Config ----------------
API_BASE = os.getenv("WESAFE_API_BASE", "http://localhost:8000")
CHAT_EP = f"{API_BASE}/chat"
HEALTH_EP = f"{API_BASE}/healthz"
RETRIEVE_EP = f"{API_BASE}/test_retriever"

# Preface di benvenuto (dopo che session_id esiste)  ----------------
TRIAGE_MODE = True
if TRIAGE_MODE and not st.session_state.preface_shown:
    try:
        print("CHECK TIRAGE_MODE")
        r = requests.post(
            CHAT_EP,
            json={"session_id": st.session_state.session_id, "message": ""},
            timeout=100
        )
        preface = (r.json() or {}).get("reply", "")
        if preface:
            print("SHOW PREFACE")
            with st.chat_message("assistant"):
                st.markdown(preface)
        st.session_state.preface_shown = True
    except Exception:
        # non bloccare la UI se il preface fallisce
        st.session_state.preface_shown = True

# ---------------- Sidebar ----------------
st.sidebar.title("WeSafe – Streamlit UI")
api_base_in = st.sidebar.text_input("API base", API_BASE)
if api_base_in != API_BASE:
    API_BASE = api_base_in
    CHAT_EP = f"{API_BASE}/chat"
    HEALTH_EP = f"{API_BASE}/healthz"
    RETRIEVE_EP = f"{API_BASE}/test_retriever"

sid = st.sidebar.text_input("Session ID", st.session_state.session_id)
st.session_state.session_id = sid.strip() or "streamlit"
print(f"Session ID: {st.session_state.session_id}")
show_docs = st.sidebar.checkbox("Mostra documenti recuperati", value=True)
print(f"Show docs: {show_docs}")


def health_status():
    try:
        r = requests.get(HEALTH_EP, timeout=10)
        j = r.json()
        return j.get("status", "unknown")
        print("CHECK HEALTH")   
    except Exception as e:
        return f"offline ({e.__class__.__name__})"

status = health_status()
st.sidebar.markdown(f"**Stato API:** `{status}`")
if status.startswith("error"):
    st.sidebar.error(status)
elif status.startswith("offline"):
    st.sidebar.warning(status)
else:
    st.sidebar.success(status)

if st.sidebar.button("Pulisci conversazione"):
    st.session_state.messages.clear()
    st.session_state.last_docs = []
    st.experimental_rerun()

# ---------------- Corpo ----------------
st.title("WeSafe – Chat notarile & visure")

with st.container():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ---------------- Input utente ----------------
prompt = st.chat_input("Scrivi qui…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("PROMPT INSERITO")
    with st.chat_message("user"):
        st.markdown(prompt)

    # /chat
    try:
        payload = {"session_id": st.session_state.session_id, "message": prompt}
        r = requests.post(CHAT_EP, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json() or {}
        # FIX: fallback se 'reply' manca
        reply = data.get("reply") or data.get("error") or str(data)
    except Exception as e:
        reply = f"⚠️ Errore chiamando l'API: `{e}`"

    # /test_retriever (non blocca)
    docs = []
    try:
        rr = requests.get(RETRIEVE_EP, params={"q": prompt}, timeout=120)
        if rr.ok:
            jr = rr.json() or {}
            docs = jr.get("docs", []) or []
    except Exception:
        pass
    st.session_state.last_docs = docs

    # assistant
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

# ---------------- Documenti recuperati ----------------
if show_docs:
    
    st.divider()
    with st.expander("📄 Documenti recuperati (top K)", expanded=True):
        docs = st.session_state.get("last_docs", [])
        if not docs:
            st.caption("Nessun documento disponibile per l’ultimo prompt.")
        else:
            for i, d in enumerate(docs, 1):
                preview = (d or "").strip()
                if len(preview) > 800:
                    preview = preview[:800] + "…"
                st.markdown(f"**Doc {i}**")
                st.code(preview)
