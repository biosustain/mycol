from pathlib import Path

import streamlit as st

DOCS_URL = "https://biosustain.github.io/mycol"
REPO_URL = "https://github.com/biosustain/mycol"
# make_dist writes the release number beside app.py; a source checkout has none.
VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"
VERSION = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "development build"

# (icon, title, description, url, button label)
GUIDES = [
    (
        "menu_book",
        "User guide",
        "Start here. Understand what Mycol does and what workflows it supports, "
        "from uploading images to exporting results.",
        f"{DOCS_URL}/index.html",
        "Open user guide",
    ),
    (
        "quiz",
        "FAQ",
        "Common questions and answers about Mycol, including troubleshooting and tips for best results.",
        f"{DOCS_URL}/faq.html",
        "Open FAQ",
    ),
    (
        "tune",
        "Functionality guide",
        "A detailed guide to the buttons and settings in the Mycol interface.",
        f"{DOCS_URL}/functionality.html",
        "Open functionality guide",
    ),
]

_, body, _ = st.columns([1, 6, 1])

with body:

    st.info(
        "**App looks too cramped or too small?** Adjust your browser's zoom.",
        icon=":material/zoom_in:",
    )

    for col, (icon, title, text, url, label) in zip(st.columns(3), GUIDES):
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"### :material/{icon}: {title}")
            st.write(text)
            # Fill the card's spare height so every button sits on the bottom edge.
            st.space("stretch")
            st.link_button(label, url, icon=":material/open_in_new:", type="primary", width="stretch")

    st.space("medium")
    st.info(
        "**If this has been helpful to you, please cite:**\n"
        "- **Mycol:** Bradley, S.A., Webel, H., Schiesaro, G. et al. Mycol: a user-friendly app for "
        "automating object identification and classification in biological images. "
        "*Appl Microbiol Biotechnol* (2026). https://doi.org/10.1007/s00253-026-14097-1\n"
        "- **Cellpose 2 (segmentation model):** Pachitariu, M., Stringer, C. Cellpose 2.0: how to train your own model. "
        "*Nat Methods* 19, 1634–1641 (2022). https://doi.org/10.1038/s41592-022-01663-4\n"
        "- **Cellpose 3 (segmentation model):** Stringer, C., Pachitariu, M. Cellpose3: one-click image restoration for "
        "improved cellular segmentation. *Nat Methods* 22, 592–599 (2025). https://doi.org/10.1038/s41592-025-02595-5\n"
        "- **DenseNet (classification model):** Huang, G., Liu, Z., Van Der Maaten, L. et al. Densely connected "
        "convolutional networks. *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, "
        "4700–4708 (2017). https://doi.org/10.1109/CVPR.2017.243",
        icon=":material/format_quote:",
    )

    st.space("medium")
    st.markdown("#### More resources")
    c1, c2 = st.columns(2)
    c1.link_button(
        "Read the paper",
        "https://doi.org/10.1007/s00253-026-14097-1",
        icon=":material/article:",
        width="stretch",
    )
    c2.link_button(
        "Report a problem on GitHub",
        f"{REPO_URL}/issues",
        icon=":material/bug_report:",
        width="stretch",
    )

    st.space("medium")
    st.markdown(f"Mycol version {VERSION}")
