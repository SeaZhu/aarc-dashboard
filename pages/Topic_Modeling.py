import streamlit as st

from app_utils import render_topics, require_processed_data


st.set_page_config(page_title="Topic Modeling", page_icon="🧩")


def main():
    st.title("Topic Modeling")
    if not require_processed_data():
        return

    render_topics(st.session_state.clean_texts)


if __name__ == "__main__":
    main()
