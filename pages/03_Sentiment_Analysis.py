import streamlit as st

from app_utils import ensure_data_with_sidebar, hide_main_nav_entry, render_sentiment, require_processed_data


st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")


def main():
    hide_main_nav_entry()
    st.title("Sentiment Analysis")
    df = ensure_data_with_sidebar()
    if df is None or df.empty:
        return
    if not require_processed_data():
        return
    render_sentiment(df, st.session_state.text_col, st.session_state.group_col, st.session_state.sentiment_df)


if __name__ == "__main__":
    main()
