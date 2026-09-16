# app.py
import os
import streamlit as st
from src.helpers.preload import eager_load_heavy_libs

PLOTLY_CONFIG = {"toImageButtonOptions": {"format": "svg"}}

# Guarded: Streamlit re-executes this script on every rerun.
if not getattr(st.plotly_chart, "_svg_default", False):

    def _svg_default(orig):
        def wrapper(figure_or_data, *args, **kwargs):
            kwargs["config"] = {**PLOTLY_CONFIG, **(kwargs.get("config") or {})}
            return orig(figure_or_data, *args, **kwargs)

        wrapper._svg_default = True
        return wrapper

    st.plotly_chart = _svg_default(st.plotly_chart)

st.set_page_config(page_title="Mycol", page_icon="👨🏼‍🔬", layout="wide")

# No icon_image: navigation is position="top", so there is no expanded
# sidebar and icon_image would win everywhere, hiding the wordmark.
st.logo("logo.png", size="large", link="https://biosustain.github.io/mycol/index.html")

# Eager load heavy libraries to prevent lag on tab switching
eager_load_heavy_libs()


# ------------------ Boot steps ------------------ #
from src.helpers.state_ops import reset_global_state_defaults

reset_global_state_defaults()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ------------------ Navigation ------------------ #
# Tab styling lives in src/helpers/nav_styles.py.
# Switch looks with: MYCOL_NAV_STYLE=<name> streamlit run app.py
from src.helpers.nav_styles import build_navigation

nav = build_navigation()

# ------------------ Run selected page ------------------ #
nav.run()
