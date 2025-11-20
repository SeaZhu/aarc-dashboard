import streamlit as st

from app_utils import render_cleaning, require_processed_data


st.set_page_config(page_title="Text Cleaning & Keywords", page_icon="🧹")


def main():
    st.title("Text Cleaning & Keywords")
    if not require_processed_data():
        return

    df = st.session_state.df
    render_cleaning(df, st.session_state.text_col, st.session_state.clean_texts, st.session_state.tokens_list)


if __name__ == "__main__":
    main()
