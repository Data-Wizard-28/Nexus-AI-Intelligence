<div align="center">

# 💠 Nexus AI — Neural PDF Intelligence Engine

### Ask questions. Get answers. See exactly where they came from.

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-ff4b4b?style=for-the-badge)](https://nexus-ai-intelligence-7iqtutwhj4cmxs6lqvlfwb.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Data--Wizard--28-181717?style=for-the-badge&logo=github)](https://github.com/Data-Wizard-28/Nexus-AI-Intelligence)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3-F55036?style=for-the-badge)](https://groq.com)

</div>

---

## What is Nexus AI?

Nexus AI is a production-grade **Retrieval-Augmented Generation (RAG)** application that lets you have a conversation with any PDF document. Unlike simple keyword search or basic RAG demos, Nexus uses a **three-stage pipeline** — hybrid search, neural re-ranking, and visual grounding — to ensure every answer is accurate, explainable, and directly traceable to the source text.

Upload a PDF, ask a question, and Nexus will:
- Find the most relevant passages using both semantic and keyword search
- Re-rank them using a neural Cross-Encoder for precision
- Generate a streamed answer using Llama 3.3 via Groq's ultra-fast LPU
- Highlight the exact source text in the PDF so you can verify every claim
- Show you a confidence score for the answer

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 **Hybrid Search** | Dense (FAISS semantic) + Sparse (BM25 keyword) retrieval combined |
| 🧠 **Neural Re-ranking** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scores and re-orders candidates |
| 📄 **PDF Visual Grounding** | Source chunks highlighted directly in the document with colour themes |
| ⚡ **Streamed Responses** | Token-by-token streaming via Groq LPU — near-instant output |
| 📊 **Confidence Score** | Sigmoid-normalised relevance score shown per answer |
| 🎨 **Highlight Themes** | Choose from Cyan, Emerald, Amber, or Rose highlight colours |
| ⬇️ **Download Highlighted PDF** | Export the annotated PDF after any query |
| 🔢 **Page Navigator** | Prev/Next navigation with the viewer jumping to the relevant page automatically |
| 🛠️ **Debug Mode** | Toggle a full JSON traceback of retrieval scores and source paths |
| 🌐 **Deployed-App Safe** | PDF viewer uses PyMuPDF rasterization — no browser PDF plugin required |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              HYBRID RETRIEVAL               │
│                                             │
│   ┌─────────────────┐  ┌─────────────────┐  │
│   │  Dense Search   │  │  Sparse Search  │  │
│   │  FAISS + MiniLM │  │  BM25 Okapi     │  │
│   └────────┬────────┘  └────────┬────────┘  │
│            └────────┬───────────┘            │
│                     ▼                        │
│           Deduplicated Candidate Pool        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              NEURAL RE-RANKING              │
│   Cross-Encoder: ms-marco-MiniLM-L-6-v2    │
│   → Top 3 chunks selected by relevance     │
└─────────────────────┬───────────────────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
┌──────────────────┐   ┌──────────────────────┐
│   LLM ANSWER     │   │   PDF HIGHLIGHTING   │
│  Llama 3.3-70B   │   │  PyMuPDF annotation  │
│  via Groq Cloud  │   │  Multi-strategy      │
│  (streamed)      │   │  phrase matching     │
└──────────────────┘   └──────────────────────┘
```

### Highlighting Engine

The PDF highlighter uses a multi-strategy approach to handle the inconsistencies between extracted text and the PDF glyph stream:

1. **Normalisation** — strips soft hyphens, non-breaking spaces, and expands ligatures (`ﬁ` → `fi`, `ﬂ` → `fl`, etc.)
2. **Progressive phrase shortening** — tries 150-char prefix → 100-char → 60-char
3. **Sentence-level fallback** — splits on `.!?` and tries each sentence longest-first
4. **Sliding window fallback** — tries every 8-word window across the chunk
5. **Adjacent page search** — tries `page ± 1` to handle PyPDFLoader off-by-one metadata errors
6. **`TEXT_DEHYPHENATE` flag** — tells PyMuPDF to join hyphenated line-breaks before matching

---

## 🖥️ UI Overview

```
┌─────────────────┬──────────────────────────────────────────┐
│    SIDEBAR      │                MAIN AREA                  │
│                 │                                           │
│  💠 NEXUS       │  ┌─────────────────┬──────────────────┐  │
│                 │  │   Status Cards  │                   │  │
│  📥 Source      │  │ Status | Memory │                   │  │
│  [File card]    │  │        | Page   │   PDF VIEWER      │  │
│  [Reset]        │  ├─────────────────┤                   │  │
│                 │  │                 │  ◀ Prev  Page N   │  │
│  🎨 Appearance  │  │   CHAT BOX      │      of M  Next ▶ │  │
│  [Theme picker] │  │                 │                   │  │
│                 │  │  👤 User msg    │  [Page rendered   │  │
│  🛠️ Developer  │  │  💠 AI answer   │   as image]       │  │
│  [Debug toggle] │  │     + pills     │                   │  │
│                 │  │                 │                   │  │
│  [⬇️ Download]  │  └─────────────────┴──────────────────┘  │
└─────────────────┴──────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com) (takes ~1 minute to get)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Data-Wizard-28/Nexus-AI-Intelligence.git
cd Nexus-AI-Intelligence
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your Groq API key**

Create a `.streamlit/secrets.toml` file:
```bash
mkdir -p .streamlit
```
```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_your_key_here"
```

Or export it as an environment variable:
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

**5. Set the theme**

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#58a6ff"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#161b22"
textColor = "#c9d1d9"
font = "sans serif"
```

**6. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploying to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In **App settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Deploy — the live app is already running at:
   👉 [nexus-ai-intelligence-7iqtutwhj4cmxs6lqvlfwb.streamlit.app](https://nexus-ai-intelligence-7iqtutwhj4cmxs6lqvlfwb.streamlit.app/)

---

## 📁 Project Structure

```
Nexus-AI-Intelligence/
│
├── app.py                  # Streamlit UI — layout, chat, PDF viewer
├── rag_pipeline.py         # Core RAG logic — search, rerank, highlight, LLM
├── requirements.txt        # Python dependencies
│
├── .streamlit/
│   ├── config.toml         # Theme configuration (primaryColor etc.)
│   └── secrets.toml        # API keys (never commit this)
│
├── data/                   # Uploaded PDFs stored here at runtime
├── highlighted.pdf         # Generated at runtime after each query
└── .gitignore
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pymupdf` (`fitz`) | PDF rendering, page rasterization, highlight annotations |
| `pypdf` | PDF text extraction via LangChain loader |
| `langchain` + `langchain-community` | RAG orchestration |
| `langchain-openai` | Groq-compatible OpenAI client wrapper |
| `langchain-huggingface` | HuggingFace embeddings integration |
| `langchain-text-splitters` | Recursive character text splitting |
| `faiss-cpu` | Dense vector similarity search |
| `rank-bm25` | BM25 sparse keyword retrieval |
| `sentence-transformers` | Cross-Encoder re-ranking model |

---

## ⚙️ Configuration

### Chunk Settings (`rag_pipeline.py`)
```python
# Adjust for your document type
chunk_size    = 700   # characters per chunk
chunk_overlap = 150   # overlap between chunks
```

### Retrieval Settings
```python
k = 5          # candidates retrieved per method (dense + sparse)
top_n = 3      # final chunks after re-ranking sent to LLM
```

### PDF Render Quality (`app.py`)
```python
dpi = 150      # increase for sharper preview, decrease for faster loading
```

### LLM Model
The default is `llama-3.3-70b-versatile` via Groq. You can swap to any Groq-supported model in `rag_pipeline.py`:
```python
model_name = "llama-3.3-70b-versatile"   # default
# model_name = "mixtral-8x7b-32768"      # alternative
# model_name = "gemma2-9b-it"            # lighter option
```

---

## 🔑 Getting a Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Go to **API Keys → Create API Key**
4. Copy the key starting with `gsk_`

Groq's free tier is generous — Llama 3.3 70B runs at hundreds of tokens per second at no cost.

---

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open source. See the repository for license details.

---

<div align="center">

Built by [Data-Wizard-28](https://github.com/Data-Wizard-28) · Powered by Groq + LangChain + Streamlit

⭐ Star the repo if you find it useful!

</div>