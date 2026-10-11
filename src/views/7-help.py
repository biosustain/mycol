import streamlit as st

DOCS_URL = "https://biosustain.github.io/mycol"
REPO_URL = "https://github.com/biosustain/mycol"

# (icon, title, description, url, button label)
GUIDES = [
    (
        "menu_book",
        "User guide",
        "Start here. Understand what Mycol does and what worklows it supports, "
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
    st.markdown("#### More resources")
    c1, c2 = st.columns(2)
    c1.link_button(
        "Read the preprint",
        "https://www.biorxiv.org/content/10.64898/2026.06.02.729113v1",
        icon=":material/article:",
        width="stretch",
    )
    c2.link_button(
        "Report a problem on GitHub",
        f"{REPO_URL}/issues",
        icon=":material/bug_report:",
        width="stretch",
    )
