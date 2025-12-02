import streamlit as st

from app_utils import ensure_data_with_sidebar, render_cleaning, require_processed_data


st.set_page_config(page_title="Text Cleaning & N-grams", page_icon="🧹")


def main():
    df = ensure_data_with_sidebar()
    if df is None or df.empty:
        return
    if not require_processed_data():
        return
    render_cleaning(df, st.session_state.text_col, st.session_state.clean_texts, st.session_state.tokens_list)


if __name__ == "__main__":
    main()
