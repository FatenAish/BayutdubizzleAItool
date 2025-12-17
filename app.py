import os
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
import os

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# ============================================================
# CHAT HISTORY STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("<h4 style='margin-top: 30px;'>Select an option</h4>", unsafe_allow_html=True)
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="text-align:center; margin-top:-30px;">
    <h1 style="font-size:42px; font-weight:700;">
        <span style="color:#0E8A6D;">Bayut</span> 
        & 
        <span style="color:#D71920;">Dubizzle</span>
        AI Content Assistant
    </h1>
    <p style="font-size:16px; color:#444; margin-top:-15px;">
        Fast internal knowledge search powered by internal content.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DATA DIRECTORY HANDLING
# ============================================================
DATA_DIR = "/app/data"
LOCAL = "./data"
if os.path.exists(LOCAL):
    DATA_DIR = LOCAL


# ============================================================
# EMBEDDINGS
# ============================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ============================================================
# VECTORSTORE PATH
# ============================================================
FAISS_DIR = os.path.join(DATA_DIR, "faiss_store")


# ============================================================
# BUILD VECTORSTORE
# ============================================================
def build_vectorstore():
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            try:
                with open(
                    os.path.join(DATA_DIR, file),
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:
                    text = f.read()

                for chunk in splitter.split_text(text):
                    documents.append(
                        Document(page_content=chunk, metadata={"source": file})
                    )
            except:
                continue

    if not documents:
        return None

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_DIR)
    return db


# ============================================================
# LOAD VECTORSTORE
# ============================================================
def load_vectorstore():
    if os.path.exists(FAISS_DIR):
        try:
            return FAISS.load_local(
                FAISS_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            return None
    return None


# ============================================================
# INIT VECTORSTORE
# ============================================================
vectorstore = load_vectorstore() or build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# ============================================================
# CLEAN ANSWER EXTRACTION — FIXED
# Removes ALL Q9/Q10/Q11, ALL question lines, leaves ONLY answer text
# ============================================================
def extract_clean_answer(raw_text):
    import re

    text = raw_text.strip()

    # Remove any Q<number> – question sentence entirely
    text = re.sub(r"Q\d+\s*–[^\.!?]*[\.!?]", "", text)

    # Remove duplicate spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# INPUT FORM (fast + prevents unwanted reruns)
# ============================================================
st.subheader(mode)

with st.form("ask_form", clear_on_submit=True):
    query = st.text_input("Ask your question:")
    submitted = st.form_submit_button("Search")


# ============================================================
# CENTERED BUTTONS
# ============================================================
colA, colB, colC, colD = st.columns([1, 1, 1, 1])

with colB:
    clear_clicked = st.button("🗑️ Clear Chat", use_container_width=True)

with colC:
    rebuild_clicked = st.button("🔄 Rebuild Index", use_container_width=True)


# ============================================================
# CLEAR CHAT — fully working
# ============================================================
if clear_clicked:
    st.session_state.history = []
    st.rerun()


# ============================================================
# REBUILD INDEX
# ============================================================
if rebuild_clicked:
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    st.success("Index rebuilt successfully!")
    st.rerun()


# ============================================================
# PROCESS QUESTION
# ============================================================
if submitted and query:
    docs = retriever.invoke(query)

    if docs:
        answer = extract_clean_answer(docs[0].page_content)
    else:
        answer = "No matching internal information found."

    st.session_state.history.append(
        {"question": query, "answer": answer}
    )


# ============================================================
# DISPLAY CHAT — NEWEST FIRST
# ============================================================
for item in reversed(st.session_state.history):
    st.markdown("### ❓ Question")
    st.write(item["question"])

    st.markdown("### ✅ Answer")
    st.write(item["answer"])

    st.markdown("---")
