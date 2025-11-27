import streamlit as st

from app_utils import render_topics, require_processed_data


st.set_page_config(page_title="04 Topic Modeling (LDA)", page_icon="🧩")


def main():
    st.title("Topic Modeling (LDA)")
    if not require_processed_data():
        return

    render_topics(st.session_state.clean_texts)


if __name__ == "__main__":
    main()
