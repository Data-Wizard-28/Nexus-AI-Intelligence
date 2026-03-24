# Nexus AI | Neural PDF Intelligence Engine 💠

An advanced **RAG (Retrieval-Augmented Generation)** pipeline designed for high-precision document intelligence. Nexus AI uses hybrid search and neural re-ranking to provide accurate answers with direct PDF highlighting.

## 🚀 Key Features
- **Hybrid Search:** Combines Semantic Search (FAISS) with Keyword Search (BM25).
- **Neural Re-ranking:** Uses a Cross-Encoder (`ms-marco-MiniLM`) to ensure the most relevant context is sent to the LLM.
- **Visual Grounding:** Automatically highlights the source text within the PDF for transparency.
- **Ultra-Fast Inference:** Powered by **Groq LPU** hardware using Llama 3.3.

## 🛠️ Tech Stack
- **Orchestration:** LangChain
- **LLM:** Llama 3.3 (via Groq Cloud)
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Frontend:** Streamlit

## 📦 Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/Data-Wizard-28/Nexus-AI-Intelligence.git](https://github.com/Data-Wizard-28/Nexus-AI-Intelligence.git)

































