import streamlit as st
import os
import base64
import fitz
import uuid
import streamlit.components.v1 as components
from rag_pipeline import setup_rag_pipeline, format_chat_history

# 1. Page Config
st.set_page_config(page_title="Nexus AI | Intelligence", layout="wide", initial_sidebar_state="expanded")

# 2. Complete UI Theme Sync
st.markdown("""
    <style>
    :root { --primary-color: #58a6ff; }
    .stApp { background-color: #0d1117; color: #c9d1d9; }

    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* --- LOGO & HEADER --- */
    .logo-container { display: flex; align-items: center; gap: 12px; padding: 5px 0 !important; }
    .logo-text {
        background: linear-gradient(120deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900; font-size: 5.0rem; letter-spacing: 1.5px; margin: 0; line-height: 1.0 !important;
    }

    /* --- SIDEBAR SPACING FIXES --- */
    .section-label {
        font-size: 0.7rem; font-weight: 700; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
        display: flex; align-items: center; gap: 6px;
        margin-bottom: 15px !important;
        padding-top: 5px;
    }

    .file-name-card {
        background: rgba(88, 166, 255, 0.05);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        padding: 12px;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
    }

    /* --- CHAT PILLS --- */
    .meta-label-container { display: flex; gap: 8px; margin-top: 12px; padding-top: 10px; border-top: 1px solid #30363d; }
    .meta-label {
        background: rgba(88, 166, 255, 0.1); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.3);
        padding: 4px 10px; border-radius: 8px; font-size: 0.75rem; font-weight: 600;
    }

    /* --- PULSE DOTS --- */
    .pulse-dot { height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }
    .pulse-green { background-color: #3fb950; box-shadow: 0 0 8px #3fb950; }
    .pulse-red { background-color: #f85149; box-shadow: 0 0 8px #f85149; }

    /* --- TOGGLE: override Streamlit red with theme blue --- */
    [data-testid="stToggle"] > div[data-checked="true"],
    [data-testid="stToggle"][aria-checked="true"] > div {
        background-color: #58a6ff !important;
    }
    /* The thumb track when ON */
    [data-testid="stToggle"] > label > div[data-testid="stToggleTrack"] {
        background-color: #58a6ff !important;
    }
    /* Broader Streamlit internal toggle selectors (covers 1.30+) */
    .st-toggle-track-checked { background-color: #58a6ff !important; }
    [class*="toggleTrack"][data-checked="true"] { background-color: #58a6ff !important; }
    /* Catch-all: any checked toggle track inside sidebar */
    [data-testid="stSidebar"] input[type="checkbox"]:checked + div,
    [data-testid="stSidebar"] input[type="checkbox"]:checked ~ div {
        background-color: #58a6ff !important;
    }
    /* Streamlit 1.35+ uses this pattern */
    div[data-baseweb="toggle"] > div:first-child[class*="checked"] {
        background-color: #58a6ff !important;
    }
    div[data-baseweb="toggle"] > div[class*="Track"]:has(~ input:checked) {
        background-color: #58a6ff !important;
    }
    /* Most reliable catch-all for BaseWeb toggle */
    [data-baseweb="toggle"] [class*="Track"] { transition: background 0.2s; }
    [data-baseweb="toggle"]:has(input:checked) [class*="Track"] {
        background-color: #58a6ff !important;
    }

    /* --- CHAT CONTAINER BORDER: override red accent with theme blue --- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #30363d !important;
    }
    /* st.container(height=...) scroll container border */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #30363d !important;
    }
    /* The actual scrollable block */
    div[data-testid="stVerticalBlock"] > div[style*="overflow"] {
        border: 1px solid #30363d !important;
        border-radius: 8px;
    }
    /* Streamlit 1.40 height-container wrapper */
    .stHeightContainer, [class*="heightContainer"] {
        border-color: #30363d !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- STATE ----------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
if "pdf_salt" not in st.session_state:
    st.session_state.pdf_salt = str(uuid.uuid4())

state_keys = {"chat": [], "rag": None, "highlights": [], "file_path": None, "current_page": 0, "last_debug": None}
for k, v in state_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- PDF PAGE RENDERER ----------------
def render_pdf_page(pdf_path: str, page_num: int, dpi: int = 150):
    """
    Rasterize a single PDF page to a PNG using PyMuPDF.
    Returns raw PNG bytes. Works on local, deployed, and HTTPS environments
    because it never relies on browser PDF plugins or base64 iframe embedding.
    """
    try:
        doc = fitz.open(pdf_path)
        page_idx = max(0, min(page_num, len(doc) - 1))
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 dpi is PDF default
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes, len(doc)
    except Exception as e:
        print(f"[PDF Render] Error: {e}")
        return None, 0

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="logo-container"><span class="logo-icon" style="font-size:1.5rem;">💠</span><p class="logo-text">NEXUS</p></div>', unsafe_allow_html=True)
    st.caption("Neural PDF Intelligence Engine")
    st.markdown("<br>", unsafe_allow_html=True)

    # SOURCE SECTION
    with st.container(border=True):
        st.markdown('<div class="section-label">📥 Source</div>', unsafe_allow_html=True)
        if st.session_state.file_path:
            fname = os.path.basename(st.session_state.file_path)
            trunc = (fname[:25] + '...') if len(fname) > 25 else fname
            st.markdown(f"""
                <div class="file-name-card">
                    <div style="display: flex; align-items: center;">
                        <span style="margin-right:10px;">📄</span>
                        <span style="color: #58a6ff; font-weight: 600; font-size: 0.85rem;">{trunc}</span>
                        <div style="margin-left:auto;"><span class="pulse-dot pulse-green"></span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 Reset Session", use_container_width=True):
                if os.path.exists("highlighted.pdf"):
                    os.remove("highlighted.pdf")
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.cache_resource.clear()
                st.rerun()
        else:
            uploaded = st.file_uploader("Upload PDF", type="pdf", key=st.session_state.uploader_key, label_visibility="collapsed")
            if uploaded:
                if os.path.exists("highlighted.pdf"):
                    os.remove("highlighted.pdf")
                os.makedirs("data", exist_ok=True)
                path = os.path.join("data", uploaded.name)
                st.session_state.file_path = path
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                with st.spinner("🧠 Indexing..."):
                    st.session_state.rag, _ = setup_rag_pipeline(path)
                    st.rerun()

    # APPEARANCE SECTION
    with st.container(border=True):
        st.markdown('<div class="section-label">🎨 Appearance</div>', unsafe_allow_html=True)
        color_map = {
            "🔵 Cyan":    (0, 0.8, 1),
            "🟢 Emerald": (0.2, 0.8, 0.2),
            "🟠 Amber":   (1, 0.7, 0),
            "🔴 Rose":    (1, 0.2, 0.4),
        }
        selected_theme = st.selectbox(
            "Highlight Theme",
            list(color_map.keys()),
            label_visibility="collapsed",
            key="highlight_theme",
        )
        highlight_color = color_map[selected_theme]

    # DEVELOPER SECTION
    with st.container(border=True):
        st.markdown('<div class="section-label">🛠️ Developer</div>', unsafe_allow_html=True)
        debug_on = st.toggle("Debug Traceback", value=False)

    # Download highlighted PDF
    if st.session_state.file_path and os.path.exists("highlighted.pdf"):
        with open("highlighted.pdf", "rb") as f:
            st.download_button(
                label="⬇️ Download Highlighted PDF",
                data=f,
                file_name="nexus_highlighted.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ---------------- MAIN DASHBOARD ----------------
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    if st.session_state.file_path:
        is_online = st.session_state.rag is not None
        status_color = "#58a6ff" if is_online else "#f85149"
        status_text = "Online" if is_online else "Offline"

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; gap: 12px; margin-bottom: 25px;">
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px; flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">Status</span>
                        <span>📡</span>
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: {status_color};">{status_text}</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px; flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">Memory</span>
                        <span>🧠</span>
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #f0f6fc;">{len(st.session_state.chat)}</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px; flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">Page</span>
                        <span>📄</span>
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #f0f6fc;">{st.session_state.current_page + 1}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    chat_box = st.container(height=600)
    with chat_box:
        if not st.session_state.file_path:
            st.info("👋 Welcome. Please upload a document.")
        for m in st.session_state.chat:
            avatar = "👤" if m["role"] == "user" else "💠"
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"], unsafe_allow_html=True)
                if m["role"] == "assistant" and "meta" in m:
                    st.markdown(f"""<div class="meta-label-container">
                        <div class="meta-label">🎯 {m['meta'].get('confidence', 0)}% Confidence</div>
                        <div class="meta-label">📄 Page {m['meta'].get('pages', [0])[0] + 1}</div>
                    </div>""", unsafe_allow_html=True)

        if debug_on and st.session_state.last_debug:
            st.divider()
            st.subheader("🔍 Debug Traceback")
            st.json(st.session_state.last_debug)

    query = st.chat_input("Query Nexus AI...")
    if query and st.session_state.rag:
        st.session_state.chat.append({"role": "user", "content": query})
        with chat_box:
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)
            with st.chat_message("assistant", avatar="💠"):
                placeholder = st.empty()
                placeholder.markdown("*⏳ Thinking...*")

                try:
                    res = st.session_state.rag(query, "", highlight_color=highlight_color)
                    st.session_state.last_debug = res

                    full_text = ""
                    first_token = True
                    for chunk in res["stream"]:
                        token = chunk if isinstance(chunk, str) else chunk.content
                        if token:
                            if first_token:
                                placeholder.empty()
                                first_token = False
                            full_text += token
                            placeholder.markdown(full_text + "▌")

                    placeholder.markdown(full_text)

                    st.session_state.chat.append({
                        "role": "assistant", "content": full_text,
                        "meta": {"confidence": res.get("confidence", 0), "pages": res.get("pages", [0])}
                    })
                    st.session_state.current_page = res["pages"][0] if res["pages"] else 0
                    st.session_state.pdf_salt = str(uuid.uuid4())
                    st.rerun()

                except Exception as e:
                    placeholder.empty()
                    st.session_state.last_debug = {"error": str(e)}
                    st.error(f"Error: {str(e)}")

# ---------------- PDF VIEWER (col2) ----------------
with col2:
    if st.session_state.file_path:
        dp = "highlighted.pdf" if os.path.exists("highlighted.pdf") else st.session_state.file_path

        # Page navigation controls
        _, total_pages = render_pdf_page(dp, 0)  # quick call just to get page count
        if total_pages > 0:
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("◀ Prev", use_container_width=True):
                    st.session_state.current_page = max(0, st.session_state.current_page - 1)
                    st.rerun()
            with nav_col2:
                st.markdown(
                    f"<div style='text-align:center; color:#8b949e; font-size:0.85rem; padding-top:8px;'>"
                    f"Page {st.session_state.current_page + 1} of {total_pages}</div>",
                    unsafe_allow_html=True
                )
            with nav_col3:
                if st.button("Next ▶", use_container_width=True):
                    st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
                    st.rerun()

            # Render current page as image — works on all platforms including HTTPS/deployed
            png_bytes, _ = render_pdf_page(dp, st.session_state.current_page, dpi=150)
            if png_bytes:
                st.markdown(
                    "<div style='border: 1px solid #30363d; border-radius: 16px; "
                    "background: #0d1117; padding: 10px; margin-top: 8px;'>",
                    unsafe_allow_html=True
                )
                st.image(png_bytes, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Could not render PDF page.")