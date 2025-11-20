import streamlit as st

from app_utils import render_sentiment, require_processed_data


st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")


def main():
    st.title("Sentiment Analysis")
    if not require_processed_data():
        return

    df = st.session_state.df
    render_sentiment(df, st.session_state.text_col, st.session_state.group_col, st.session_state.sentiment_df)


if __name__ == "__main__":
    main()
