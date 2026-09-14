"""Swappable looks for the top page navigation.

Pick one with the MYCOL_NAV_STYLE env var, or edit DEFAULT_STYLE below:

    MYCOL_NAV_STYLE=underline streamlit run app.py

Each entry in STYLES is a dict with:
    icons  -- ":material/...:" per page, or None for text-only tabs
    css    -- the <style> block injected into the app header
"""

import os

import streamlit as st

DEFAULT_STYLE = "numbered_pills"

# (path, title, material icon name)
PAGE_SPECS = [
    ("src/views/2-upload-data.py", "Upload Data", "upload_file"),
    ("src/views/3-create-and-edit-masks.py", "Annotate Images", "draw"),
    ("src/views/4-fine-tune-models.py", "Train Models", "model_training"),
    ("src/views/5-cell-metrics.py", "Compare Phenotypes", "analytics"),
    ("src/views/6-downloads.py", "Downloads", "download"),
]

# Palette pulled from .streamlit/config.toml so the tabs stay on-theme.
NAVY = "#004280"
TINT = "#E9F2FF"
MUTED = "#5B7FA6"
LINE = "#C9DCEF"
NAVY_RGB = "0, 66, 128"  # NAVY as rgb components, for translucent shadows

# --------------------------------------------------------------------------- #
# Shared add-on: the soft shadow divider between the tabs and the page body,
# as the app had before the restyle. Appended after a style's own CSS (so it
# wins on source order) for any style flagged "divider": True.
# --------------------------------------------------------------------------- #
FADED_DIVIDER_CSS = f"""
.stAppHeader {{
    /* A soft shadow divider rather than a hairline, tinted with the theme
       navy instead of black. Paints over the page body, so no z-index
       trouble. background-image clears any gradient a style set. */
    border-bottom: none !important;
    background-image: none !important;
    box-shadow:
        0 8px 28px rgba({NAVY_RGB}, 0.22),
        0 3px 8px rgba({NAVY_RGB}, 0.12) !important;
}}
"""

# --------------------------------------------------------------------------- #
# classic -- the look the app shipped with: tinted bar, bold 20px labels.
# --------------------------------------------------------------------------- #
CLASSIC_CSS = f"""
.stAppHeader {{
    background-color: {TINT} !important;
    padding: 12px 20px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.2) !important;
}}
.stAppHeader span, .stAppHeader div {{
    font-weight: 600 !important;
}}
.stAppHeader span, .stAppHeader h1, .stAppHeader div {{
    font-size: 20px !important;
}}
"""

# --------------------------------------------------------------------------- #
# underline -- flat white bar, active page marked by a thick navy underline.
# --------------------------------------------------------------------------- #
UNDERLINE_CSS = f"""
.stAppHeader {{
    background-color: #ffffff !important;
    padding: 0 24px !important;
    box-shadow: none !important;
    border-bottom: 1px solid {LINE} !important;
}}
[data-testid="stTopNavLink"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    padding: 16px 2px !important;
    margin: 0 16px !important;
    transition: border-color 0.15s ease, color 0.15s ease;
}}
[data-testid="stTopNavLink"] span {{
    color: {MUTED} !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
}}
[data-testid="stTopNavLink"]:hover {{
    border-bottom-color: {LINE} !important;
}}
[data-testid="stTopNavLink"]:hover span {{
    color: {NAVY} !important;
}}
[data-testid="stTopNavLink"][aria-current="page"] {{
    border-bottom-color: {NAVY} !important;
}}
[data-testid="stTopNavLink"][aria-current="page"] span {{
    color: {NAVY} !important;
    font-weight: 700 !important;
}}
"""

# --------------------------------------------------------------------------- #
# pills -- active page is a filled navy pill; hover previews with a pale tint.
# --------------------------------------------------------------------------- #
PILLS_CSS = f"""
.stAppHeader {{
    background-color: #ffffff !important;
    padding: 10px 24px !important;
    box-shadow: none !important;
    border-bottom: 1px solid {LINE} !important;
}}
[data-testid="stTopNavLink"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 9px 18px !important;
    margin: 0 4px !important;
    transition: background-color 0.15s ease, color 0.15s ease;
}}
[data-testid="stTopNavLink"] span {{
    color: {MUTED} !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}}
[data-testid="stTopNavLink"]:hover {{
    background: {TINT} !important;
}}
[data-testid="stTopNavLink"]:hover span {{
    color: {NAVY} !important;
}}
[data-testid="stTopNavLink"][aria-current="page"] {{
    background: {NAVY} !important;
    box-shadow: 0 1px 3px rgba(0,66,128,0.30) !important;
}}
[data-testid="stTopNavLink"][aria-current="page"] span {{
    color: #ffffff !important;
}}
"""

# --------------------------------------------------------------------------- #
# stepper -- the five pages are a linear pipeline, so number them and chain
# them with chevrons. Badges come from a CSS counter on the link containers.
# --------------------------------------------------------------------------- #
STEPPER_CSS = f"""
.stAppHeader {{
    background-color: #FAFCFF !important;
    padding: 8px 24px !important;
    box-shadow: none !important;
    border-bottom: 1px solid {LINE} !important;
    counter-reset: navstep;
}}
.stAppHeader [data-testid="stTopNavLinkContainer"] {{
    counter-increment: navstep;
    display: inline-flex !important;
    align-items: center !important;
}}
/* chevron between steps, but not after the last one */
.stAppHeader [data-testid="stTopNavLinkContainer"]:not(:last-child)::after {{
    content: "\203A";
    color: {LINE};
    font-size: 20px;
    line-height: 1;
    margin: 0 2px;
}}
.stAppHeader [data-testid="stTopNavLink"] {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    transition: background-color 0.15s ease;
}}
/* the numbered badge */
.stAppHeader [data-testid="stTopNavLink"]::before {{
    content: counter(navstep);
    flex: 0 0 auto;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 1.5px solid {LINE};
    color: {MUTED};
    background: #ffffff;
    font-size: 12px;
    font-weight: 700;
    line-height: 19px;
    text-align: center;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}}
.stAppHeader [data-testid="stTopNavLink"] span {{
    color: {MUTED} !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover {{
    background: {TINT} !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover span {{
    color: {NAVY} !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"]::before {{
    background: {NAVY};
    border-color: {NAVY};
    color: #ffffff;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] span {{
    color: {NAVY} !important;
    font-weight: 700 !important;
}}
"""

# --------------------------------------------------------------------------- #
# icons -- Material icon stacked above a small label, app-launcher style.
# Uses the icon names already listed in PAGE_SPECS.
# --------------------------------------------------------------------------- #
ICONS_CSS = f"""
.stAppHeader {{
    background-color: #ffffff !important;
    padding: 0 24px !important;
    box-shadow: none !important;
    border-bottom: 1px solid {LINE} !important;
}}
.stAppHeader [data-testid="stTopNavLink"] {{
    display: inline-flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 3px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 8px 14px !important;
    margin: 0 2px !important;
    min-width: 86px !important;
    transition: background-color 0.15s ease;
}}
.stAppHeader [data-testid="stTopNavLink"] [data-testid="stIconMaterial"] {{
    font-size: 24px !important;
    width: 24px !important;
    height: 24px !important;
    color: {MUTED} !important;
    transition: color 0.15s ease;
}}
.stAppHeader [data-testid="stTopNavLink"] span {{
    color: {MUTED} !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    text-align: center !important;
    line-height: 1.25 !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover {{
    background: {TINT} !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover span,
.stAppHeader [data-testid="stTopNavLink"]:hover [data-testid="stIconMaterial"] {{
    color: {NAVY} !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] {{
    background: {TINT} !important;
    box-shadow: inset 0 -3px 0 0 {NAVY} !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] span,
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{
    color: {NAVY} !important;
    font-weight: 700 !important;
}}
"""

# --------------------------------------------------------------------------- #
# folder -- classic file-folder tabs. Inactive tabs sit recessed in a tinted
# bar; the active tab turns white and overlaps the header border by 1px so it
# merges into the page body below.
# --------------------------------------------------------------------------- #
FOLDER_CSS = f"""
.stAppHeader {{
    background-color: #EDF3FA !important;
    padding: 12px 24px 0 24px !important;
    box-shadow: none !important;
    border-bottom: 1px solid {LINE} !important;
}}
.stAppHeader [data-testid="stTopNavLinkContainer"] {{
    align-self: flex-end !important;
}}
.stAppHeader [data-testid="stTopNavLink"] {{
    background: #DDE9F6 !important;
    border: 1px solid {LINE} !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 11px 18px !important;
    /* the -1px pulls the tab down over the header's bottom border */
    margin: 0 3px -1px 3px !important;
    transition: background-color 0.15s ease;
}}
.stAppHeader [data-testid="stTopNavLink"] span {{
    color: {MUTED} !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover {{
    background: {TINT} !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover span {{
    color: {NAVY} !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] {{
    background: #ffffff !important;
    box-shadow: inset 0 3px 0 0 {NAVY} !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] span {{
    color: {NAVY} !important;
    font-weight: 700 !important;
}}
"""

# --------------------------------------------------------------------------- #
# sections -- structural rather than cosmetic: the five pages collapse into one
# plain link plus two dropdown groups, so the bar carries three items instead of
# five. GROUPS below maps a section label to indices into PAGE_SPECS; the ""
# key holds ungrouped pages, which Streamlit always renders first.
# --------------------------------------------------------------------------- #
SECTION_GROUPS = {
    "": [0],
    "Segment & Train": [1, 2],
    "Analyze": [3, 4],
}

SECTIONS_CSS = f"""
.stAppHeader {{
    background-color: #ffffff !important;
    padding: 8px 24px !important;
    box-shadow: none !important;
    border-bottom: 1px solid {LINE} !important;
}}
.stAppHeader [data-testid="stTopNavLink"],
.stAppHeader [data-testid="stTopNavSection"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin: 0 4px !important;
    transition: background-color 0.15s ease;
}}
.stAppHeader [data-testid="stTopNavLink"] span,
.stAppHeader [data-testid="stTopNavSection"] span {{
    color: {MUTED} !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover,
.stAppHeader [data-testid="stTopNavSection"]:hover {{
    background: {TINT} !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover span,
.stAppHeader [data-testid="stTopNavSection"]:hover span {{
    color: {NAVY} !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] {{
    background: {TINT} !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] span {{
    color: {NAVY} !important;
    font-weight: 700 !important;
}}
/* the dropdown panel renders in a portal, so it sits outside .stAppHeader */
[data-testid="stTopNavPopover"] {{
    border: 1px solid {LINE} !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 24px rgba(0,66,128,0.12) !important;
    padding: 6px !important;
}}
[data-testid="stTopNavDropdownLink"] {{
    border-radius: 6px !important;
    padding: 9px 14px !important;
}}
[data-testid="stTopNavDropdownLink"] span {{
    color: {MUTED} !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}}
[data-testid="stTopNavDropdownLink"]:hover {{
    background: {TINT} !important;
}}
[data-testid="stTopNavDropdownLink"]:hover span {{
    color: {NAVY} !important;
}}
[data-testid="stTopNavDropdownLink"][aria-current="page"] {{
    background: {TINT} !important;
}}
[data-testid="stTopNavDropdownLink"][aria-current="page"] span {{
    color: {NAVY} !important;
    font-weight: 700 !important;
}}
"""

# --------------------------------------------------------------------------- #
# numbered_pills -- pills, plus the stepper's numbered badges. Active page is a
# filled navy pill whose badge inverts to white; no chevrons, the numbers alone
# carry the running order.
# --------------------------------------------------------------------------- #
NUMBERED_PILLS_CSS = f"""
/* Reclaim the right-hand toolbar's width for the labels. The column holding
   the Deploy button and the main menu carries a hard min-width: 200px while
   its real content is a 32px menu button, so hiding Deploy alone frees
   nothing -- the min-width has to go too. */
[data-testid="stAppDeployButton"] {{
    display: none !important;
}}
.stAppHeader div:has(> [data-testid="stToolbarActions"]) {{
    min-width: auto !important;
}}
.stAppHeader {{
    background-color: #ffffff !important;
    padding: 10px 24px !important;
    box-shadow: none !important;
    border-bottom: 1px solid {LINE} !important;
    counter-reset: navstep;
}}
.stAppHeader [data-testid="stTopNavLinkContainer"] {{
    counter-increment: navstep;
}}
/* Give every pill an equal share of the bar. Streamlit wraps each item in an
   rc-overflow-item, then its own div, so the flex list sits three levels above
   a link container. Matched via :has() rather than the .rc-overflow / hashed
   emotion classes, which are not a stable API. The :has() on the item also
   excludes rc-overflow's empty "N more" placeholder, which holds no link and
   so must not claim a share of the width. */
/* The header row is space-between, so width freed by hiding Deploy would
   become a gap unless the list itself grows into it. */
.stAppHeader div:has(> div > div > [data-testid="stTopNavLinkContainer"]) {{
    flex-grow: 1 !important;
}}
.stAppHeader div:has(> div > div > [data-testid="stTopNavLinkContainer"])
    > div:has(> div > [data-testid="stTopNavLinkContainer"]) {{
    flex: 1 1 0 !important;
    min-width: 0 !important;
}}
.stAppHeader [data-testid="stTopNavLinkContainer"] {{
    width: 100% !important;
}}

.stAppHeader [data-testid="stTopNavLink"] {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 999px !important;
    width: 100% !important;
    justify-content: center !important;
    padding: 8px 12px 8px 8px !important;
    margin: 0 2px !important;
    transition: background-color 0.15s ease;
}}
.stAppHeader [data-testid="stTopNavLink"]::before {{
    content: counter(navstep);
    flex: 0 0 auto;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 1.5px solid {LINE};
    background: #ffffff;
    color: {MUTED};
    font-size: 13px;
    font-weight: 700;
    line-height: 21px;
    text-align: center;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}}
.stAppHeader [data-testid="stTopNavLink"] span {{
    color: {MUTED} !important;
    font-size: 19px !important;
    font-weight: 600 !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover {{
    background: {TINT} !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover span {{
    color: {NAVY} !important;
}}
.stAppHeader [data-testid="stTopNavLink"]:hover::before {{
    border-color: {NAVY};
    color: {NAVY};
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] {{
    background: {NAVY} !important;
    box-shadow: 0 1px 3px rgba(0,66,128,0.30) !important;
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"]::before {{
    background: #ffffff;
    border-color: #ffffff;
    color: {NAVY};
}}
.stAppHeader [data-testid="stTopNavLink"][aria-current="page"] span {{
    color: #ffffff !important;
}}
"""

STYLES = {
    "classic": {"icons": False, "css": CLASSIC_CSS},
    "underline": {"icons": False, "css": UNDERLINE_CSS},
    "pills": {"icons": False, "css": PILLS_CSS},
    "stepper": {"icons": False, "css": STEPPER_CSS},
    "icons": {"icons": True, "css": ICONS_CSS},
    "folder": {"icons": False, "css": FOLDER_CSS},
    "sections": {"icons": False, "css": SECTIONS_CSS, "groups": SECTION_GROUPS},
    "numbered_pills": {
        "icons": False,
        "css": NUMBERED_PILLS_CSS,
        "divider": True,
    },
}


def _style_name() -> str:
    name = os.environ.get("MYCOL_NAV_STYLE", DEFAULT_STYLE)
    return name if name in STYLES else DEFAULT_STYLE


def build_navigation():
    """Inject the chosen tab styling and return the st.navigation object."""
    style = STYLES[_style_name()]
    css = style["css"]
    if style.get("divider"):
        css += FADED_DIVIDER_CSS
    st.html(f"<style>{css}</style>")

    pages = [
        st.Page(
            path,
            title=title,
            icon=f":material/{icon}:" if style["icons"] else None,
            default=(i == 0),
        )
        for i, (path, title, icon) in enumerate(PAGE_SPECS)
    ]

    groups = style.get("groups")
    if groups:
        pages = {label: [pages[i] for i in idx] for label, idx in groups.items()}

    return st.navigation(pages, position="top", expanded=False)
