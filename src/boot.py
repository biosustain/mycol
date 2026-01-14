# boot.py (new small helper)
import streamlit as st


@st.cache_resource(show_spinner=False)
def configure_tf_cpu_only():
    # Deprecated, removed tf from dependencies
    pass


def common_boot():
    # st.set_page_config(page_title="Mask Toggle", layout="wide")

    # make main section buttons white
    st.markdown(
        """
    <style>
    /* === Global material-style buttons (main + sidebar) === */
    div.stButton > button,
    div.stDownloadButton > button,
    section[data-testid="stSidebar"] div.stButton > button,
    section[data-testid="stSidebar"] div.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #004280 !important;                /* your navy */
    border: 1px solid rgba(0,66,128,0.25) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    font-weight: 500 !important;
    transition: transform .06s ease, box-shadow .12s ease, background-color .12s ease;
    }

    /* Hover: lift slightly + deepen shadow */
    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    section[data-testid="stSidebar"] div.stButton > button:hover,
    section[data-testid="stSidebar"] div.stDownloadButton > button:hover {
    background-color: #F8FAFE !important;     /* subtle tint */
    box-shadow: 0 4px 14px rgba(0,0,0,0.10) !important;
    transform: translateY(-1px);
    }

    /* Active/pressed: dip a bit + compress shadow */
    div.stButton > button:active,
    div.stDownloadButton > button:active,
    section[data-testid="stSidebar"] div.stButton > button:active,
    section[data-testid="stSidebar"] div.stDownloadButton > button:active {
    background-color: #F2F6FB !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
    transform: translateY(0);
    }

    /* Focus ring (keyboard accessibility) */
    div.stButton > button:focus,
    div.stDownloadButton > button:focus,
    section[data-testid="stSidebar"] div.stButton > button:focus,
    section[data-testid="stSidebar"] div.stDownloadButton > button:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(0,66,128,0.35), 0 4px 12px rgba(0,0,0,0.10) !important;
    }

    /* Make container-width buttons look nice */
    div.stButton > button[kind],
    div.stDownloadButton > button[kind] {
    width: 100%;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # make sidebar buttons white
    st.markdown(
        """
    <style>
    /* ===== Apply to ALL buttons (main + sidebar) ===== */
    div.stButton > button,
    section[data-testid="stSidebar"] div.stButton > button,
    div.stDownloadButton > button,
    section[data-testid="stSidebar"] div.stDownloadButton > button {
        background-color: white !important;
        color: #004280 !important;
        border: 1.5px solid #004280 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out;
    }

    /* ===== Hover state ===== */
    div.stButton > button:hover,
    section[data-testid="stSidebar"] div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    section[data-testid="stSidebar"] div.stDownloadButton > button:hover {
        background-color: #F2F6FB !important;  /* subtle light-blue tint */
        color: #004280 !important;
        border-color: #004280 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
