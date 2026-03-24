import os
import re
import math
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def format_chat_history(chat, max_turns=3):
    """Formats chat history for context-aware queries."""
    history = chat[-max_turns * 2:]
    return "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history])


# ---------------------------------------------------------------------------
# Highlighting helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Collapse all whitespace (spaces, newlines, tabs, non-breaking spaces,
    soft-hyphens) into a single ASCII space and strip leading/trailing space.
    This is the single most important step — PDF text extracted by PyMuPDF /
    PyPDF has random newlines and extra spaces baked in.
    """
    text = text.replace('\u00ad', '')    # soft hyphen
    text = text.replace('\u00a0', ' ')   # non-breaking space
    text = text.replace('\ufb00', 'ff')  # ff ligature
    text = text.replace('\ufb01', 'fi')  # fi ligature
    text = text.replace('\ufb02', 'fl')  # fl ligature
    text = text.replace('\ufb03', 'ffi')
    text = text.replace('\ufb04', 'ffl')
    text = re.sub(r'[ \t]+', ' ', text)  # collapse horizontal space
    text = re.sub(r'\n+', ' ', text)     # newlines → space
    return text.strip()


def _candidate_phrases(text: str):
    """
    Yield progressively shorter search candidates from a chunk of text.
    Order: longest-first so we get the most precise highlight possible,
    then fall back gracefully.

    Strategy:
      1. First 150 chars normalised
      2. First 100 chars normalised
      3. First 60 chars normalised
      4. Every sentence (split on .!?) that is >= 30 chars, longest first
      5. Every 8-word sliding window across the text (catches mid-chunk matches)
    """
    norm = _normalize(text)

    # Fixed-length prefixes
    for length in (150, 100, 60):
        candidate = norm[:length].strip()
        if len(candidate) >= 20:
            yield candidate

    # Sentence-level fallback
    sentences = re.split(r'(?<=[.!?])\s+', norm)
    for s in sorted(sentences, key=len, reverse=True):
        s = s.strip()
        if len(s) >= 30:
            yield s

    # Sliding-window fallback (8 words)
    words = norm.split()
    if len(words) >= 8:
        for i in range(0, len(words) - 7, 3):  # step=3 keeps it fast
            window = ' '.join(words[i:i + 8])
            if len(window) >= 20:
                yield window


def _search_page(page, phrase: str):
    """
    Try fitz search with TEXT_DEHYPHENATE flag so PyMuPDF joins hyphenated
    line-breaks before matching. Falls back to plain search_for if the flag
    variant finds nothing or raises (older PyMuPDF versions).
    """
    try:
        hits = page.search_for(phrase, quads=True, flags=fitz.TEXT_DEHYPHENATE)
    except Exception:
        hits = []

    if not hits:
        try:
            hits = page.search_for(phrase, quads=True)
        except Exception:
            hits = []

    return hits


def apply_highlights(source_pdf: str, highlights: list,
                     output_pdf: str = "highlighted.pdf",
                     color: tuple = (1, 0.8, 0)):
    """
    Robustly highlight every chunk in `highlights` on its source page.

    For each chunk we try multiple candidate phrases (longest-first) until
    at least one hit is found on the page. If the exact page has no hit we
    also try the adjacent pages (+/-1) to handle off-by-one metadata errors
    that PyPDFLoader sometimes produces.
    """
    if not os.path.exists(source_pdf):
        print(f"[Highlight] Source PDF not found: {source_pdf}")
        return

    try:
        doc = fitz.open(source_pdf)
        total_pages = len(doc)

        for item in highlights:
            raw_page = item.get("page", 0)
            text_to_find = item.get("text", "")
            if not text_to_find.strip():
                continue

            # Pages to try: declared page first, then +/-1 neighbours
            pages_to_try = [raw_page]
            if raw_page > 0:
                pages_to_try.append(raw_page - 1)
            if raw_page < total_pages - 1:
                pages_to_try.append(raw_page + 1)

            highlighted_this_chunk = False

            for pg_idx in pages_to_try:
                if pg_idx < 0 or pg_idx >= total_pages:
                    continue
                page = doc[pg_idx]

                for phrase in _candidate_phrases(text_to_find):
                    hits = _search_page(page, phrase)
                    if hits:
                        for inst in hits:
                            annot = page.add_highlight_annot(inst)
                            annot.set_colors(stroke=color)
                            annot.update()
                        highlighted_this_chunk = True
                        break   # stop trying shorter phrases for this page

                if highlighted_this_chunk:
                    break       # stop trying neighbour pages

            if not highlighted_this_chunk:
                print(f"[Highlight] No match found for chunk starting: "
                      f"{text_to_find[:60]!r}")

        doc.save(output_pdf)
        doc.close()
        print(f"[Highlight] Saved -> {output_pdf}")

    except Exception as e:
        print(f"[Highlight] Fatal error: {e}")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    return splitter.split_documents(docs)


def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_llm():
    """
    Uses Groq Cloud (OpenAI-compatible) to run Llama for free.
    Set GROQ_API_KEY in your environment or Streamlit Secrets.
    """
    return ChatOpenAI(
        model_name="llama-3.3-70b-versatile",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1"
    )


def create_bm25_index(chunks):
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    return BM25Okapi(tokenized)


def create_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def normalize_scores(scores, query, context):
    if not scores:
        return 0
    max_logit = max(min(max(scores), 10), -10)
    sig_score = 1 / (1 + math.exp(-max_logit))

    query_words = set(re.findall(r'\w+', query.lower()))
    context_words = set(re.findall(r'\w+', context.lower()))
    if query_words.intersection(context_words) and sig_score < 0.1:
        sig_score = 0.45

    return int(min(sig_score, 0.99) * 100)


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

def create_rag_chain(llm, db, chunks, bm25, reranker, original_file_path):
    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY based on context. If unknown, say so.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    def rag_pipeline(question, history, highlight_color=(1, 0.8, 0)):
        # 1. Hybrid Search (Dense + Sparse)
        dense = db.similarity_search(question, k=5)
        bm25_scores = bm25.get_scores(question.lower().split())
        sparse = [
            chunks[i]
            for i in sorted(range(len(bm25_scores)),
                            key=lambda i: bm25_scores[i], reverse=True)[:5]
        ]

        unique_docs = []
        seen = set()
        for d in dense + sparse:
            if d.page_content not in seen:
                unique_docs.append(d)
                seen.add(d.page_content)

        # 2. Reranking
        pairs = [(question, d.page_content) for d in unique_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)

        top_docs   = [d for d, s in ranked[:3]]
        top_scores = [float(s) for d, s in ranked[:3]]
        context_text = "\n\n".join([d.page_content for d in top_docs])

        # 3. Highlighting
        highlights = [
            {"text": d.page_content, "page": d.metadata.get("page", 0)}
            for d in top_docs
        ]

        if os.path.exists("highlighted.pdf"):
            try:
                os.remove("highlighted.pdf")
            except Exception:
                pass

        apply_highlights(original_file_path, highlights, color=highlight_color)

        return {
            "stream": llm.stream(
                prompt.invoke({"context": context_text, "question": question})
            ),
            "pages": sorted(set(d.metadata.get("page", 0) for d in top_docs)),
            "confidence": normalize_scores(top_scores, question, context_text),
            "debug_info": {
                "top_scores": top_scores,
                "source_path": original_file_path,
            },
        }

    return rag_pipeline


def setup_rag_pipeline(file_path):
    docs   = load_document(file_path)
    chunks = split_documents(docs)
    db     = FAISS.from_documents(chunks, create_embeddings())
    return (
        create_rag_chain(
            create_llm(), db, chunks,
            create_bm25_index(chunks), create_reranker(),
            file_path
        ),
        len(chunks),
    )