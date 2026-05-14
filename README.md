# 🤖 Multi-Session RAG Chatbot with PDF Support
 
A conversational AI chatbot built with **Streamlit**, **LangChain**, and **Groq LLM** that supports multiple chat sessions and optional PDF-based Retrieval-Augmented Generation (RAG).
 
---
 
## ✨ Features
 
- 💬 **Multi-session chat** — maintain separate conversation histories simultaneously
- 📄 **PDF RAG** — upload one or more PDFs and ask questions grounded in their content
- 🧠 **Context-aware retrieval** — history-aware retriever rewrites queries before searching
- 🔍 **Source transparency** — sidebar shows the top retrieved document chunks with page numbers
- ⚡ **Fast embeddings** — powered by `all-MiniLM-L6-v2` via HuggingFace
- 🗂️ **FAISS vector store** — efficient local similarity search, rebuilt only when uploads change
---
 
## 🛠️ Tech Stack
 
| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`Gemma2-9b-It`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Orchestration | LangChain |
| PDF Loading | PyPDFLoader |
 
---
 
## 🚀 Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/Avi9482/Multi-Session-RAG-Integrated-chatbot.git
cd Avi9482
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Set up environment variables
 
Create a file named `a.env` in the project root:
 
```env
GROQ_API_KEY=your_groq_api_key_here
```
 
> Get your free Groq API key at [console.groq.com](https://console.groq.com)
 
### 4. Run the app
 
```bash
streamlit run final_x.py
```
 
---
 
## 📁 Project Structure
 
```
.
├── final_x.py        # Main Streamlit application
├── utils.py          # Helper utilities
├── a.env             # Environment variables (not committed)
├── requirements.txt  # Python dependencies
└── README.md
```
 
---
 
## 🧭 How It Works
 
### Without PDF
The chatbot runs as a standard conversational assistant. It maintains per-session message history so context is preserved across turns.
 
### With PDF(s) Uploaded
1. Uploaded PDFs are temporarily saved, loaded via `PyPDFLoader`, and split into chunks.
2. Chunks are embedded and stored in a FAISS vector store.
3. On each query, a **history-aware retriever** first rewrites the question as a standalone query using chat history, then retrieves the top-3 relevant chunks.
4. The LLM answers using **only** the retrieved context.
---
 
## 🖥️ Usage
 
1. **Start a session** — a default session (`newchat`) is created automatically.
2. **Create new sessions** — click "New session" in the sidebar to start a fresh conversation.
3. **Switch sessions** — use the dropdown to jump between existing sessions (remove any uploaded files first).
4. **Upload PDFs** — drag and drop one or more PDF files to enable RAG mode.
5. **Ask questions** — type in the input box and hit Submit.
6. **Inspect sources** — expand "Top retrieved chunks" in the sidebar to see which pages were used.
---
 
## ⚙️ Configuration
 
| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 1200 | Characters per text chunk |
| `chunk_overlap` | 100 | Overlap between consecutive chunks |
| `k` (retriever) | 3 | Number of chunks retrieved per query |
| `temperature` | 0.2 | LLM response randomness |
| `max_tokens` | 512 | Maximum tokens in LLM response |
 
---
 
## 📋 Requirements
 
```
streamlit
langchain
langchain-community
langchain-huggingface
langchain-groq
faiss-cpu
pypdf
python-dotenv
sentence-transformers
```
 
---
 
## 🔒 Notes
 
- The `a.env` file containing your API key should **never** be committed to version control. Add it to `.gitignore`.
- The vector store is rebuilt only when the set of uploaded files changes, avoiding redundant reprocessing.
- Switching sessions or creating a new session while a PDF is uploaded is intentionally blocked — remove the file first to prevent context bleed between sessions.
---
 
## 📄 License
 
This project is licensed under the [MIT License](LICENSE).
 
---
 
## 🙌 Acknowledgements
 
- [LangChain](https://www.langchain.com/) for the chain and retriever abstractions
- [Groq](https://groq.com/) for ultra-fast LLM inference
- [HuggingFace](https://huggingface.co/) for open-source embeddings
- [Streamlit](https://streamlit.io/) for the rapid UI framework
