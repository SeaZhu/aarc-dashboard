import streamlit as st

from app_utils import ensure_data_with_sidebar, hide_main_nav_entry, render_topics, require_processed_data


st.set_page_config(page_title="04 Topic Modeling (LDA)", page_icon="🧩")


def main():
    hide_main_nav_entry()
    st.title("Topic Modeling (LDA)")
    df = ensure_data_with_sidebar()
    if df is None or df.empty:
        return
    if not require_processed_data():
        return
    render_topics(st.session_state.clean_texts)


if __name__ == "__main__":
    main()
