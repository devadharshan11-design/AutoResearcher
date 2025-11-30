import os
import shutil
import time
from typing import List

import streamlit as st

from core.orchestrator import (
    build_index_from_pdfs,
    answer_question_with_rag,
    multi_agent_answer,
)
from config import VECTOR_DB_DIR

DATA_PDF_DIR = os.path.join("data", "pdfs")
os.makedirs(DATA_PDF_DIR, exist_ok=True)


# ================== UTILS ==================


def save_uploaded_pdfs(uploaded_files) -> List[str]:
    saved_paths = []
    for file in uploaded_files:
        if file is None:
            continue
        filename = os.path.basename(file.name)
        save_path = os.path.join(DATA_PDF_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(file.read())
        saved_paths.append(save_path)
    return saved_paths


def list_existing_pdfs() -> List[str]:
    if not os.path.exists(DATA_PDF_DIR):
        return []
    return [f for f in os.listdir(DATA_PDF_DIR) if f.lower().endswith(".pdf")]


def clear_index(index_name: str):
    index_dir = os.path.join(VECTOR_DB_DIR, index_name)
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [{role, content, mode, index_name}]


# ================== MAIN APP ==================


def main():
    st.set_page_config(
        page_title="AutoResearcher",
        page_icon="🧠",
        layout="wide",
    )

    init_session_state()

    # ---------- Custom CSS ----------
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .stApp {
            background: radial-gradient(circle at top left, #151b2b, #050608 55%);
            color: #f5f5f5;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 0.95rem;
            color: #a0a6b8;
            margin-bottom: 1.5rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.12);
            color: #e2e8f0;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }

        .card {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .section-caption {
            font-size: 0.85rem;
            color: #a0a6b8;
            margin-bottom: 0.6rem;
        }

        .stButton>button {
            border-radius: 999px;
            padding: 0.45rem 1.2rem;
            border: none;
            font-weight: 600;
            background: linear-gradient(135deg, #f97316, #ec4899);
            color: white;
            box-shadow: 0 12px 30px rgba(249, 115, 22, 0.30);
        }
        .stButton>button:hover {
            filter: brightness(1.1);
        }

        .chat-bubble-user {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            border-radius: 18px 18px 4px 18px;
            padding: 0.6rem 0.8rem;
            color: white;
            font-size: 0.94rem;
            margin-bottom: 0.4rem;
        }
        .chat-bubble-bot {
            background: rgba(15, 23, 42, 0.96);
            border-radius: 18px 18px 18px 4px;
            padding: 0.6rem 0.8rem;
            color: #e5e7eb;
            font-size: 0.94rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            margin-bottom: 0.4rem;
        }
        .chat-meta {
            font-size: 0.7rem;
            color: #9ca3af;
            margin-bottom: 0.2rem;
        }

        .index-pill {
            display: inline-flex;
            font-size: 0.72rem;
            padding: 0.1rem 0.5rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.16);
            color: #e5e7eb;
            margin-left: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Top header ----------
    st.markdown(
        """
        <div class="main-title">
            <span>🧠 AutoResearcher</span>
        </div>
        <div class="subtitle">
            <span class="pill">Multi-Agent Research Assistant</span>
            <span class="pill">Local LLM (Ollama)</span>
            <span class="pill">GPU Embeddings · RAG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Sidebar ----------
    st.sidebar.header("⚙️ Settings")

    default_index_name = "edge_ai_paper"
    index_name = st.sidebar.text_input(
        "Index name",
        value=default_index_name,
        help="Same index name = same combined knowledge base of PDFs.",
    )

    mode = st.sidebar.radio(
        "Answering mode",
        ["Multi-agent (Searcher + Critic + Writer)", "Simple RAG"],
        help="Multi-agent chains 3 roles; Simple RAG = one-step answer.",
    )

    top_k = st.sidebar.slider(
        "Chunks to retrieve (top_k)",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "💻 **Backend:** Ollama LLM + SentenceTransformers + Local Vector Store"
    )
    st.sidebar.markdown("⚠️ Make sure **Ollama** is running.")

    # ---------- Main layout: left (docs), right (chat) ----------
    col_docs, col_chat = st.columns([1.4, 2.0])

    # ================= LEFT: DOCUMENTS =================
    with col_docs:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header">📂 Manage Documents & Index</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Upload PDFs. All PDFs built under the same index name are searched together.</div>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Drag & drop or browse PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("📚 Build / Update Index", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF first.")
            else:
                with st.spinner("Saving PDFs and building index..."):
                    pdf_paths = save_uploaded_pdfs(uploaded_files)
                    build_index_from_pdfs(pdf_paths, index_name=index_name)
                st.success(
                    f"Index **'{index_name}'** updated with {len(uploaded_files)} file(s)."
                )

        st.markdown("---", unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header">📄 Current PDFs in Library</div>',
            unsafe_allow_html=True,
        )
        pdfs = list_existing_pdfs()
        if not pdfs:
            st.write("No PDFs found yet.")
        else:
            for name in pdfs:
                st.markdown(f"- 🧾 `{name}`")

        st.markdown("---", unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header">🧹 Index Maintenance</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear This Index"):
            clear_index(index_name)
            st.success(
                f"Index **'{index_name}'** cleared. PDFs stay; only embeddings are removed."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ================= RIGHT: CHAT =================
    with col_chat:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header">💬 Ask the Research Assistant</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Questions are answered using only the papers in the selected index.</div>',
            unsafe_allow_html=True,
        )

        # Show chat history
        for msg in st.session_state.chat_history:
            role = msg["role"]
            bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
            who = "You" if role == "user" else "Assistant"
            mode_text = msg.get("mode", "")
            idx_text = msg.get("index_name", "")

            st.markdown(
                f'<div class="chat-meta">{who}'
                f'<span class="index-pill">{mode_text} · index: {idx_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="{bubble_class}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        user_question = st.chat_input("Type your research question here...")

        if user_question:
            # Store user message
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": user_question,
                    "mode": mode,
                    "index_name": index_name,
                }
            )

            # Answer
            with st.chat_message("assistant"):
                if mode.startswith("Multi-agent"):
                    st.markdown(
                        f"🧠 *Multi-agent mode* — Searcher + Critic + Writer  \n"
                        f"Index: `{index_name}` • top_k = {top_k}"
                    )
                    with st.spinner("Thinking across agents..."):
                        t0 = time.time()
                        result = multi_agent_answer(
                            question=user_question,
                            index_name=index_name,
                            top_k=top_k,
                        )
                        t1 = time.time()

                    st.success(f"Done in {t1 - t0:.1f} seconds.")
                    st.markdown("#### ✅ Final Answer")
                    st.write(result["final_answer"])

                    with st.expander("🔍 Searcher Agent Summary"):
                        st.write(result["searcher_summary"])

                    with st.expander("🧪 Critic Agent Feedback"):
                        st.write(result["critic_feedback"])

                    final_text = result["final_answer"]

                else:
                    st.markdown(
                        f"📄 *Simple RAG mode*  \nIndex: `{index_name}` • top_k = {top_k}"
                    )
                    with st.spinner("Retrieving chunks and generating answer..."):
                        t0 = time.time()
                        answer = answer_question_with_rag(
                            question=user_question,
                            index_name=index_name,
                            top_k=top_k,
                        )
                        t1 = time.time()

                    st.success(f"Done in {t1 - t0:.1f} seconds.")
                    st.markdown("#### ✅ Answer")
                    st.write(answer)
                    final_text = answer

            # Store assistant message
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": final_text,
                    "mode": mode,
                    "index_name": index_name,
                }
            )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
