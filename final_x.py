# ================== IMPORTING LIBRARIES ==================
import os
import uuid

import streamlit as st


from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from utils import show_top_chunks_with_page
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# ================== MODELS ==================
load_dotenv(dotenv_path="a.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_API_KEY=st.secrets['GROQ_API_KEY']

# Guard API key
if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY. Please add it to a.env")
    st.stop()

# LLM + Embeddings
llm = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=512,
)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
st.title("BIG Chuchi BIG 🍑 Chatbot 💦")
st.write("Multi‑Session Chat with Optional PDF RAG powered by LangChain’s message history")

# ================== STATE INIT ==================
if "store" not in st.session_state:
    st.session_state.store = {}         # LC histories per session_id
if "disp" not in st.session_state:
    st.session_state.disp = {}          # UI transcripts per session_id
if "messages_display" not in st.session_state:
    st.session_state.messages_display = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "newchat"
if "newchat_counter" not in st.session_state:
    st.session_state.newchat_counter = 1
if "files_hash" not in st.session_state:
    st.session_state.files_hash = None  # to rebuild KB only when uploads change

# ================== UTILS ==================
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

def _is_invalid_session_id(sid: str) -> bool:
    return (sid is None) or (not isinstance(sid, str)) or (sid.strip() == "")


def _generate_new_session_id() -> str:
    sid = f"newchat_{st.session_state.newchat_counter:03d}"
    st.session_state.newchat_counter += 1
    return sid


def bind_messages_display_to_current_session():
    """Ensure messages_display points to the transcript list of the active session."""
    sid = st.session_state.session_id
    if _is_invalid_session_id(sid):
        sid = _generate_new_session_id()
        st.session_state.session_id = sid
    if sid not in st.session_state.disp:
        st.session_state.disp[sid] = []
    st.session_state.messages_display = st.session_state.disp[sid]

def clear_vector_db():
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.files_hash = None



# # ================== HELPERS ==================
def extract_answer(raw):
    """Normalize any chain output to a displayable string."""
    # if raw is None:
    #     return ""
    # if isinstance(raw, dict):
    #     return raw.get("answer") or raw.get("text") or raw.get("output") or ""
    content = getattr(raw, "content", None)
    if content is not None:
        return content
    to_dict = getattr(raw, "model_dump", None) or getattr(raw, "dict", None)
    if callable(to_dict):
        # try:
            d = to_dict()
            return d.get("answer") or d.get("text") or d.get("output") or d.get("content") or ""
    #     except Exception:
    #         return ""
    return str(raw)



def switch_session(new_sid: str):
    """Switch active session and rebind transcript; ensure LC history exists."""
    st.session_state.user_input = ""
    if _is_invalid_session_id(new_sid):
        return
    st.session_state.session_id = new_sid
    if new_sid not in st.session_state.disp:
        st.session_state.disp[new_sid] = []
    st.session_state.messages_display = st.session_state.disp[new_sid]
    if new_sid not in st.session_state.store:
        st.session_state.store[new_sid] = ChatMessageHistory()

    clear_vector_db()
    return True

def create_new_session():
    new_sid = _generate_new_session_id()
    st.session_state.disp[new_sid] = []
    st.session_state.store[new_sid] = ChatMessageHistory()
    switch_session(new_sid)


def show_top_chunks_with_page(raw, title="Top retrieved sources", max_chunks=3):
    # raw is expected to be a dict with "context" = list[Document]
    if(st.session_state.retriever==None):
      return
    docs = raw.get("context", []) if isinstance(raw, dict) else []
    with st.sidebar.expander(title, expanded=True):
        if not docs:
            st.caption("No retrieved chunks.")
            return
        # Take exactly the first N chunks returned by the retriever/chain
        for i, doc in enumerate(docs[:max_chunks], 1):
            meta = getattr(doc, "metadata", {}) or {}
            page = meta.get("page") or meta.get("page_number")
            content = getattr(doc, "page_content", "") or ""
            # Header shows chunk index and source page number
            header = f"Chunk {i} — Page {page}" if page is not None else f"Chunk {i} — Page ?"
            st.markdown(f"**{header}**")
            st.code(content)

bind_messages_display_to_current_session()

# ================== UPLOAD & INDEX ==================
uploaded_files = st.file_uploader(
    "Choose PDF file(s) (optional)",
    type="pdf",
    accept_multiple_files=True,
)


def compute_files_hash(files) -> str | None:
    if not files:
        return None
    names = tuple(sorted(f.name for f in files))
    return str(names)

# Only rebuild when the set of files changes
current_hash = compute_files_hash(uploaded_files)
rebuild = current_hash is not None and current_hash != st.session_state.files_hash
if current_hash is None:
    clear_vector_db()
if rebuild:

    documents = []
    for uf in uploaded_files:
        temp_name = f"./tmp_{uuid.uuid4().hex}.pdf"
        with open(temp_name, "wb") as f:
            f.write(uf.getvalue())
        docs = PyPDFLoader(temp_name).load()
        documents.extend(docs)
        os.remove(temp_name)


    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    st.session_state.vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    st.session_state.files_hash = current_hash
    st.success(f"Indexed {len(splits)} chunks from {len(uploaded_files)} file(s).")

# elif current_hash is None:
#     # No files: do not reset vectorstore/retriever; just leave them as-is
#     pass


# ================== SIDEBAR CONTROLS ==================
def get_options():
    keys = list(st.session_state.disp.keys())
    if st.session_state.session_id not in keys:
        keys.append(st.session_state.session_id)
    return sorted(set(keys))

with st.sidebar:
    st.write(f"Current session: {st.session_state.session_id}")

    # Session switcher
    options = get_options()
    selected = st.selectbox(
        "Select session",
         options=options,
         index=options.index(st.session_state.session_id) if st.session_state.session_id in options else 0,
          key="session_selector",
    )
    if selected != st.session_state.session_id:
        if not uploaded_files:
          switch_session(selected)
          st.success(f"Switched to: {selected}")
        else:
          st.error('First delete uploaded file!')

    if st.button("New session"):
        if not uploaded_files:
          create_new_session()
          st.rerun()
          st.success(f"Created: {st.session_state.session_id}")
        else:
          st.error('First delete uploaded file!')


    with st.expander("Debug: Chat history", expanded=True):
        hist = get_session_history(st.session_state.session_id)
        st.caption(f"History length: {len(hist.messages)/2}")
        show_preview = st.checkbox("Show last 3 conversations", value=False)
        if show_preview:
            recent = hist.messages[-6:]
            for i, m in enumerate(reversed(recent), 1):
                role = getattr(m, "type", getattr(m, "role", "message"))
                content = getattr(m, "content", str(m))
                snippet = content if len(content) <= 200 else content[:200] + "..."
                st.markdown(f"- {i}. [{role}] {snippet}")
    # st.write(f'{current_hash}')

# ================== CHAIN ==================
def rag_chain(user_input):
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rewrite the latest question as standalone using chat history. Do not answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([

        ("system",
        "The user has uploaded a PDF. Use ONLY the retrieved context from that PDF to answer the question.\n\n"
        "Answer using the retrieved context. If unknown, say only 'i don't know'. \n\ncontext:{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    core_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    output_key = "answer"
    conversational_chain = RunnableWithMessageHistory(
        core_chain,
         get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key=output_key,
    )

    res=conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )
    show_top_chunks_with_page(res, title="Top retrieved chunks")
    # return core_chain,'answer'
    return res['answer']


def chain(user_input):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Chat normally, considering prior messages."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # core_chain = LLMChain(llm=llm, prompt=chat_prompt)
    core_chain=chat_prompt | llm
    output_key = "output"
    conversational_chain = RunnableWithMessageHistory(
        core_chain,
         get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key=output_key,
    )

    res=conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
    )
    
    # return core_chain,'text'
    return res.content

def main_chain(user_input):
  if st.session_state.retriever is None:
    answer=chain(user_input)
  else:
    answer=rag_chain(user_input)
  return answer


# ================== CHAT UI ==================
st.divider()

with st.form("ask"):
    user_input = st.text_input("Your question:", key="user_input")
    submitted = st.form_submit_button("Submit")


if submitted and user_input:
    answer=main_chain(user_input)
    # Store user turn (newest-first UI)
    st.session_state.messages_display.insert(0, (user_input, ""))
    st.session_state.messages_display[0] = (user_input,answer)
    # show_top_chunks_with_page(answer, title="Top retrieved chunks")


# Render transcript (newest first)
for user_msg, bot_msg in st.session_state.messages_display:
    st.write(f"**You:** {user_msg}")
    st.write(f"**Assistant:** \n {bot_msg}")
    st.write("---")

