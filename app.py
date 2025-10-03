import os
import requests
import streamlit as st

st.set_page_config(page_title="WeSafe – Chat", page_icon="🗂️", layout="centered")

# ---------------- Stato ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit"
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "preface_shown" not in st.session_state:
    st.session_state.preface_shown = False  # default esplicito

# ---------------- Config ----------------
API_BASE = os.getenv("WESAFE_API_BASE", "http://localhost:8000")
CHAT_EP = f"{API_BASE}/chat"
HEALTH_EP = f"{API_BASE}/healthz"
CLEAR_CHAT = f"{API_BASE}/reset_session"

# Preface di benvenuto (solo la prima volta)
TRIAGE_MODE = True
if TRIAGE_MODE and not st.session_state.preface_shown:
    try:
        r = requests.post(
            CHAT_EP,
            json={"session_id": st.session_state.session_id, "message": ""},
            timeout=30,
        )
        preface = (r.json() or {}).get("reply", "")
        if preface:
            with st.chat_message("assistant"):
                st.markdown(preface)
        st.session_state.preface_shown = True
    except Exception:
        st.session_state.preface_shown = True  # non bloccare la UI

# ---------------- Sidebar ----------------
st.sidebar.title("WeSafe – Streamlit UI")
api_base_in = st.sidebar.text_input("API base", API_BASE)
if api_base_in != API_BASE:
    API_BASE = api_base_in
    CHAT_EP = f"{API_BASE}/chat"
    HEALTH_EP = f"{API_BASE}/healthz"
    CLEAR_CHAT = f"{API_BASE}/reset_session"

sid = st.sidebar.text_input("Session ID", st.session_state.session_id)
st.session_state.session_id = sid.strip() or "streamlit"

def health_status():
    try:
        r = requests.get(HEALTH_EP, timeout=5)
        j = r.json()
        return j.get("status", "unknown")  
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
    # reset lato UI
    st.session_state.messages.clear()
    st.session_state.last_docs = []

    # reset lato central agent (se implementato)
    try:
        requests.post(CLEAR_CHAT, params={"session_id": st.session_state.session_id}, timeout=10)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Errore nel reset central agent: {e}")

# ---------------- Corpo ----------------
st.title("WeSafe Assistant")

with st.container():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ---------------- Input utente ----------------
prompt = st.chat_input("Scrivi qui…")
if prompt:
    # aggiungi messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # invia al central agent
    try:
        payload = {"session_id": st.session_state.session_id, "message": prompt}
        r = requests.post(CHAT_EP, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json() or {}
        reply = data.get("reply") or data.get("error") or str(data)
    except Exception as e:
        reply = f"⚠️ Errore chiamando l'API: `{e}`"

    # mostra risposta
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
