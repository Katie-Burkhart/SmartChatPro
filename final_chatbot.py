"""
Smart Chat Pro (RAG Edition)
Enhanced for Task 4: Handles long conversations, no results, and graceful fallbacks.
Now includes prompt-injection detection and sanitization.
"""

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from database import ChatDatabase
from auth import AuthManager
import uuid
from datetime import datetime
import json
import tiktoken
import traceback

# === Import RAG + Security Utils ===
from utils.rag import (
    fuse_dense_bm25,
    answer_with_context,
    rerank_chunks,
    rewrite_query,
    init_chroma
)
from utils.security import (
    is_on_topic,
    is_assignment_intent,
    contains_assignment_docs,
    SAFE_ASSIGNMENT_REPLY,
    OFF_TOPIC_REPLY,
    is_prompt_injection,
    sanitize_user_input,
    chunks_contain_injection
)

# === Load env & initialize clients ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db = ChatDatabase()
auth = AuthManager()

# === Page config ===
st.set_page_config(
    page_title="Smart Chat Pro (RAG)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Session State Initialization ===
for key, default in {
    "authenticated": False,
    "user_id": None,
    "username": None,
    "current_session_id": None,
    "messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# === Helpers ===
def count_tokens(messages, model_name="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model_name)
    return sum(len(enc.encode(m["content"])) for m in messages)


def debug_log(msg):
    st.markdown(
        f"<div style='background:#f6f8fa;padding:6px;border-left:3px solid #0366d6;font-size:0.9rem;'>🧩 {msg}</div>",
        unsafe_allow_html=True,
    )
    print(f"[DEBUG] {msg}")


# === AUTH ===
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🤖 Smart Chat Pro (RAG Edition)")
        st.markdown("### Learn Python the smart way — grounded, safe, and interactive.")
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    ok, uid = auth.authenticate_user(u, p)
                    if ok:
                        st.session_state.update(
                            {"authenticated": True, "user_id": uid, "username": u}
                        )
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")

        with tab2:
            with st.form("register_form"):
                ru = st.text_input("Username")
                reml = st.text_input("Email (optional)")
                rp1 = st.text_input("Password", type="password")
                rp2 = st.text_input("Confirm Password", type="password")
                reg = st.form_submit_button("Register")
                if reg:
                    if not ru or not rp1:
                        st.error("Username and password required.")
                    elif rp1 != rp2:
                        st.error("Passwords do not match.")
                    elif len(rp1) < 6:
                        st.error("Password must be ≥ 6 characters.")
                    else:
                        ok, msg = auth.create_user(ru, rp1, reml)
                        st.success("Account created! Please login.") if ok else st.error(
                            msg
                        )

# === MAIN APP ===
else:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        if st.button("🚪 Logout", use_container_width=True):
            for k in [
                "authenticated",
                "user_id",
                "username",
                "current_session_id",
                "messages",
            ]:
                st.session_state[k] = None if k != "authenticated" else False
            st.rerun()

        st.divider()
        try:
            col = init_chroma()
            st.markdown("### 🧠 Vector DB")
            st.caption(f"Collection: {col.name}")
            st.caption(f"Documents: {col.count()}")
        except Exception as e:
            st.error("Vector DB not loaded.")
            st.code(str(e))

        st.divider()
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            sid = str(uuid.uuid4())
            sname = f"Chat {datetime.now().strftime('%b %d, %H:%M')}"
            db.create_session(sid, st.session_state.user_id, sname)
            st.session_state.current_session_id = sid
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful Python tutor."}
            ]
            st.rerun()

    # === Chat Area ===
    if st.session_state.current_session_id:
        for m in st.session_state.messages[1:]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user_input = st.chat_input("Ask me about Python...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            db.add_message(st.session_state.current_session_id, "user", user_input)

            with st.chat_message("user"):
                st.write(user_input)

            tokens = count_tokens(st.session_state.messages)
            MAX_TOKENS = 100  # you can temporarily set this to 100 for testing (10000)

            # --- Token Usage Progress Bar ---
            progress = min(tokens / MAX_TOKENS, 1.0)  # ensure it doesn't exceed 100%

            if progress < 0.5:
                st.success(f"Token usage: {tokens:,}/{MAX_TOKENS:,} ({progress*100:.1f}%) ✅")
            elif progress < 0.8:
                st.warning(f"Token usage: {tokens:,}/{MAX_TOKENS:,} ({progress*100:.1f}%) ⚠️")
            else:
                st.error(f"Token usage: {tokens:,}/{MAX_TOKENS:,} ({progress*100:.1f}%) 🚨")


            st.progress(progress)

            # Display textual info
            st.caption(f"🧮 Tokens used: {tokens:,} / {MAX_TOKENS:,}  ({progress*100:.1f}%)")


            if tokens > MAX_TOKENS:
                st.warning(
                    "⚠️ This chat has become quite long. Start a new session to maintain performance."
                )
                if st.button("Start New Chat"):
                    sid = str(uuid.uuid4())
                    sname = f"Chat {datetime.now().strftime('%b %d, %H:%M')}"
                    db.create_session(sid, st.session_state.user_id, sname)
                    st.session_state.current_session_id = sid
                    st.session_state.messages = [
                        {"role": "system", "content": "You are a helpful Python tutor."}
                    ]
                    st.rerun()
                st.stop()

            # === RAG PROCESS ===
            with st.chat_message("assistant"):
                with st.spinner("Retrieving from course materials..."):
                    try:
                        debug_log("Initializing Chroma client...")
                        col = init_chroma()
                        if col.count() == 0:
                            raise ValueError("Vectorstore is empty!")

                        # --- Step 1: Prompt-injection detection ---
                        inj, reason = is_prompt_injection(user_input)
                        if inj:
                            response_text = (
                                "⚠️ I detected potentially unsafe or malicious instructions "
                                "and will not forward them to the LLM.\n\n"
                                f"(Reason: {reason})"
                            )
                            st.write(response_text)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response_text}
                            )
                            db.add_message(
                                st.session_state.current_session_id,
                                "assistant",
                                response_text,
                            )
                            st.stop()

                        # --- Step 2: Sanitize user input ---
                        user_input_clean = sanitize_user_input(user_input)

                        # --- Step 3: On-topic / off-topic ---
                        if not is_on_topic(user_input_clean):
                            response_text = OFF_TOPIC_REPLY
                        else:
                            rewritten = rewrite_query(user_input_clean)
                            debug_log(f"Rewritten query: {rewritten}")

                            candidates = fuse_dense_bm25(col, rewritten, k=8)
                            if not candidates:
                                st.warning("No results found in your course materials.")
                                response_text = (
                                    "I couldn’t find relevant material for your question in our course documents. "
                                    "Try one of these options:\n"
                                    "1️⃣ Rephrase your question using specific Python terms (e.g., 'for loop')\n"
                                    "2️⃣ Start a new chat to explore a different topic\n"
                                    "3️⃣ Ask for available modules or lessons.\n\n"
                                    "_(No relevant sources found — results limited by available PDFs.)_"
                                )
                            else:
                                top_chunks = rerank_chunks(user_input_clean, candidates, topn=3)

                                # --- Step 4: Assignment safety check ---
                                if is_assignment_intent(user_input_clean) or contains_assignment_docs(top_chunks):
                                    response_text = SAFE_ASSIGNMENT_REPLY

                                # --- Step 5: Chunk-level injection detection ---
                                else:
                                    attack, reason = chunks_contain_injection(top_chunks)
                                    if attack:
                                        response_text = (
                                            "⚠️ Some course documents contain unsafe or system-like instructions. "
                                            "I won’t use those sources.\n\n"
                                            f"(Reason: {reason})"
                                        )
                                    elif not top_chunks:
                                        response_text = (
                                            "I couldn’t find relevant material for your question. "
                                            "Try rephrasing or specify a Python topic like loops or NumPy."
                                        )
                                    else:
                                        # --- Step 6: Final Answer Generation ---
                                        response_text = answer_with_context(
                                            user_input_clean, top_chunks
                                        )

                        st.write(response_text)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_text}
                        )
                        db.add_message(
                            st.session_state.current_session_id,
                            "assistant",
                            response_text,
                        )

                    except Exception as e:
                        err = f"RAG Error: {e}"
                        st.error(err)
                        traceback.print_exc()

            st.rerun()

    else:
        st.title("🤖 Smart Chat Pro (RAG Edition)")
        st.info("👈 Start your first chat to explore RAG-powered learning!")
