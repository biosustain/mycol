from pathlib import Path
import base64
import streamlit as st
# ---------- Config ----------
SVG_DIR = Path("src/intro_images")
HELP_DIR = Path("src/help_texts")
TEXT_SIDECAREXT = (".md", ".txt")

# Maps SVG filename -> Markdown filename in HELP_DIR
TEXT_MAPPING: dict[str, str] = {
    "1_welcome_image.svg": "welcome.md",
    "2_workflow.svg": "workflow.md",
    "3_uploads.svg": "upload.md",
    "4_masks.svg": "segment_classify.md",
    "5_train.svg": "train.md",
    "6_analyze.svg": "analyze.md",
}


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def list_svgs() -> list[str]:
    exts = (".svg", ".SVG")
    return sorted([str(p) for p in SVG_DIR.glob("*") if p.suffix in exts])


@st.cache_data(show_spinner=False)
def read_svg_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def next_index(i: int, n: int) -> int:
    return (i + 1) % n if n else 0


def prev_index(i: int, n: int) -> int:
    return (i - 1 + n) % n if n else 0


def get_text_for(svg_path: str) -> str:
    """Return text for the given SVG path using mapping to markdown files."""
    p = Path(svg_path)
    
    # 1. Try explicit mapping
    md_name = TEXT_MAPPING.get(p.name) or TEXT_MAPPING.get(p.stem)
    
    if md_name:
        md_path = HELP_DIR / md_name
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")
            
    # 2. Try sidecar in SVG dir (fallback)
    for ext in TEXT_SIDECAREXT:
        sc = p.with_suffix(ext)
        if sc.exists():
            return sc.read_text(encoding="utf-8")
            
    return "No help text found."


# ---------- Optional: app-wide frame styling ----------
st.markdown(
    """
    <style>
    .stApp { background: #f6f7f9; }
    .stApp .main .block-container {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0,0,0,.12);
        padding: 24px 24px 28px 24px;
    }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load ----------
files = list_svgs()
if "idx" not in st.session_state:
    st.session_state.idx = 0
if not files:
    st.error("No SVG files found in `intro_images/`.")
    st.stop()

with st.spinner("Loading Home Page..."):
    col0, col1, col2 = st.columns([0.1, 1, 2])

    # ---------- Title + Controls ----------
    with col1:
        with st.container(border=True, height=680):
            # determine current slide
            current_path = files[st.session_state.idx]

            # display text first
            with st.container(border=False, height=560):
                st.markdown(get_text_for(current_path))

            # then buttons (visually below text, but still inside bordered container)
            col_a, col_b = st.columns(2)
            if col_a.button("⟵ Prev", width='stretch'):
                st.session_state.idx = prev_index(st.session_state.idx, len(files))
                st.rerun()
            if col_b.button("Next ⟶", width='stretch'):
                st.session_state.idx = next_index(st.session_state.idx, len(files))
                st.rerun()

    # ---------- SVG Viewer ----------
    with col2:
        current_path = files[st.session_state.idx]
        data_uri = read_svg_data_uri(current_path)
        st.markdown(
            f"""
            <div style="position:relative;max-width:960px;margin:0 auto;">
                <img
                    src="{data_uri}"
                    alt="{Path(current_path).name}"
                    style="width:100%;height:auto;max-height:80vh;
                           object-fit:contain;border-radius:12px;
                           box-shadow:0 4px 18px rgba(0,0,0,.12);"
                />
            </div>
            """,
            unsafe_allow_html=True,
        )
