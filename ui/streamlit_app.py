"""Streamlit chat UI for DocuChat. Talks to the FastAPI backend.

Run:  streamlit run ui/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("DOCUCHAT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
    except requests.ConnectionError:
        st.error(f"Can't reach the API at {API_URL}. Is `uvicorn app.api:app` running?")
        st.stop()
    if not r.ok:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        st.error(f"Error {r.status_code}: {detail}")
        return None
    return r.json()


# ---------- Sidebar: documents ----------
with st.sidebar:
    st.title("📄 DocuChat")
    st.caption("Chat with your PDFs, with page-level citations.")

    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Index documents", type="primary", use_container_width=True):
        for f in uploaded:
            with st.spinner(f"Indexing {f.name}..."):
                res = api("POST", "/documents", files={"file": (f.name, f.getvalue(), "application/pdf")})
            if res:
                st.success(f"{res['filename']}: {res['chunks']} chunks")

    st.divider()
    st.subheader("Indexed documents")
    docs = (api("GET", "/documents") or {}).get("documents", [])
    if not docs:
        st.caption("Nothing indexed yet.")
    for name in docs:
        c1, c2 = st.columns([5, 1])
        c1.write(f"• {name}")
        if c2.button("✕", key=f"del-{name}", help=f"Remove {name}"):
            api("DELETE", f"/documents/{name}")
            st.rerun()

    st.divider()
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    show_chunks = st.toggle("Show retrieved chunks", value=False)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(msg: dict):
    if msg.get("citations"):
        with st.expander(f"📚 Sources ({len(msg['citations'])})"):
            for c in msg["citations"]:
                st.markdown(f"**{c['source']}**, page {c['page']}")
                st.caption(f"“{c['quote']}”")
    if show_chunks and msg.get("retrieved"):
        with st.expander("🔎 Retrieved chunks"):
            for c in msg["retrieved"]:
                st.markdown(f"**{c['source']}** p.{c['page']} · score {c['score']}")
                st.text(c["text"][:600])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg)

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = api("POST", "/ask", json={"question": question, "top_k": top_k})
        if res:
            msg = {
                "role": "assistant",
                "content": res["answer"],
                "citations": res["citations"],
                "retrieved": res["retrieved"],
            }
            st.markdown(msg["content"])
            render_sources(msg)
            st.session_state.messages.append(msg)
