import os
import streamlit as st
from utils import load_documents, split_docs, embed_list, create_index, search, generate_answer

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "2"

st.set_page_config(page_title="KnowledgeBase Agent", page_icon="📘", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "chunks" not in st.session_state:
    st.session_state.chunks = None

st.markdown("""
<style>


html, body, [data-testid="stAppViewContainer"] {
    background: #eef2f7;
    font-family: "Inter", sans-serif;
}
section[data-testid="stSidebar"] {
    background: radial-gradient(circle at top, #0f1a32, #0b1120 60%);
    padding: 28px;
    color: #f8fafc !important;
    border-right: 1px solid rgba(255,255,255,0.06);
    box-shadow: 6px 0 25px rgba(0,0,0,0.4);
    min-width: 300px;
}
.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 14px;
    color: #ffffff !important;
}
div[data-testid="stFileUploader"] > div:first-child {
    border-radius: 18px !important;
    border: 2px dashed rgba(148,163,184,0.45) !important;
    background: #ffffff !important;
    padding: 32px 20px !important;
    text-align: center !important;
    transition: 0.25s ease-in-out;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
div[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.35);
    transform: scale(1.02);
}
div[data-testid="stFileUploader"] p {
    color: #1e293b !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    margin-bottom: 6px !important;
}
div[data-testid="stFileUploader"] small {
    color: #64748b !important;
    font-size: 13px !important;
}
div[data-testid="stFileUploader"] button {
    background: #e5efff !important;
    border-radius: 12px !important;
    color: #1e40af !important;
    border: 1px solid #bfdbfe !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    margin-top: 14px !important;
    transition: 0.2s ease-in-out !important;
}
div[data-testid="stFileUploader"] button:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
}
div[data-testid="stFileUploader"] section {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
div[data-testid="stFileUploader"] svg {
    fill: #3b82f6 !important;
}
.doc-item {
    width: 100% !important;
    background: #ffffff !important;
    padding: 14px 18px !important;
    margin-top: 16px !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    color: #1e293b !important;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    justify-content: space-between;
}
.doc-item:hover {
    transform: translateY(-3px);
    background: #f3f4f6 !important;
}
.doc-item::before {
    content: "📄";
    font-size: 20px;
}
.processed-pill {
    display: inline-block;
    margin-top: 8px;
    padding: 6px 12px;
    background: #e7edf3;
    color: #1e3a8a;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    font-size: 12px;
    font-weight: 500;
}
.indexing-title {
    margin-top: 18px !important;
    margin-bottom: 8px !important;
    color: #cbd5e1 !important;
    font-size: 14px !important;
}
div[data-testid="stFileUploader"] progress,
div[data-testid="stFileUploader"] div[data-testid*="progress"],
div[role="progressbar"],
progress {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    height: 8px !important;
    padding: 0 !important;
}
[data-testid="stProgress"] > div > div {
    background: #3b82f6 !important;
    height: 8px !important;
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%;
    margin-top: 22px;
    background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
    color: white !important;
    padding: 12px 20px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.18);
}
.stButton > button:hover {
    transform: translateY(-3px);
}
.main-box {
    max-width: 900px;
    margin: auto;
}
.header-box {
    background: white;
    padding: 28px;
    border-radius: 20px;
    margin-top: 20px;
    margin-bottom: 26px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(15,23,42,0.04);
}
.header-title {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
}
.header-sub {
    font-size: 13px;
    color: #6b7280;
}
.chat-panel {
    max-height: 500px;
    overflow-y: auto;
}
.message {
    display: flex;
    align-items: flex-start;
    margin-bottom: 16px;
}
.avatar-ai {
    width: 36px;
    height: 36px;
    background: url("https://i.imgur.com/8Km9tLL.png");
    background-size: cover;
    border-radius: 50%;
    margin-right: 12px;
}
.avatar-user {
    width: 36px;
    height: 36px;
    background: url("https://i.imgur.com/Vz81GEl.png");
    background-size: cover;
    border-radius: 50%;
    margin-left: 12px;
}
.bot-bubble {
    max-width: 72%;
    background: white;
    padding: 16px 20px;
    border-radius: 20px 20px 20px 8px;
    font-size: 15px;
    color: #1f2937;
    box-shadow: 0 6px 18px rgba(15,23,42,0.04);
}
.user-bubble {
    max-width: 72%;
    background: #cfe2ff;
    padding: 16px 20px;
    border-radius: 20px 20px 8px 20px;
    color: #0f3c89;
    margin-left: auto;
    box-shadow: 0 4px 12px rgba(2,6,23,0.06);
}
.input-container {
    margin-top: 18px;
    display: flex;
    padding: 10px 12px;
}
.msg-input {
    flex: 1;
    padding: 14px 20px;
    border-radius: 50px;
    background: #ffffff;
    font-size: 15px;
    border: 1px solid #e6eef8;
}
.send-btn {
    background: #3b82f6 !important;
    color: white !important;
    padding: 12px 26px !important;
    border-radius: 50px !important;
    margin-left: 12px !important;
}
.chat-panel::-webkit-scrollbar {
    width: 10px;
}
.chat-panel::-webkit-scrollbar-thumb {
    background: rgba(0,0,0,0.12);
    border-radius: 6px;
}
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small {
    color: #ffffff !important;
    opacity: 1 !important;
}
label[data-testid="stFileUploaderLabel"] {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sidebar-title'>Document Sources</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload (PDF, DOCX, TXT)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    if uploaded:
        os.makedirs("data", exist_ok=True)
        for f in uploaded:
            with open(os.path.join("data", f.name), "wb") as file:
                file.write(f.getbuffer())
            st.markdown(f"<div class='doc-item'>{f.name}</div>", unsafe_allow_html=True)
        st.markdown("<div class='indexing-title'>Indexing Status</div>", unsafe_allow_html=True)
        st.progress(100)

        docs = load_documents()
        chunks = split_docs(docs)
        vectors = embed_list(chunks)
        index = create_index(vectors)

        st.session_state.index_ready = True
        st.session_state.index = index
        st.session_state.chunks = chunks

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("<div class='main-box'>", unsafe_allow_html=True)

st.markdown("""
<div class='header-box'>
    <div class='header-title'>KnowledgeBase Agent </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="message" style="justify-content: flex-end;">
                <div class="user-bubble">{msg['content']}</div>
                <div class="avatar-user"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="message">
                <div class="avatar-ai"></div>
                <div class="bot-bubble">{msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

if "message_key" not in st.session_state:
    st.session_state.message_key = 0

st.markdown("<div class='input-container'>", unsafe_allow_html=True)

query = st.text_input(
    "",
    key=f"input_{st.session_state.message_key}",
    placeholder="Ask a question..."
)

send = st.button("Send", key="send_btn")

st.markdown("</div>", unsafe_allow_html=True)

if send:
    if not st.session_state.index_ready:
        st.error("Upload and process documents first.")
    elif query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("AI Thinking..."):
            ctx_chunks, score = search(query, st.session_state.index, st.session_state.chunks)
            ctx = "\n\n".join(ctx_chunks)
            ans = generate_answer(ctx, query, "Professional", st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "model", "content": ans})
        st.session_state.message_key += 1
        st.rerun()
