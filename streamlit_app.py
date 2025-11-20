from typing import Optional

import pandas as pd
import streamlit as st

from app_utils import (
    get_sentiment,
    load_data,
    preprocess_text,
)


st.set_page_config(
    page_title="Text Analytics App - AARC Survey",
    page_icon="🧠",
    layout="wide",
)


def init_state():
    defaults = {
        "df": pd.DataFrame(),
        "text_col": None,
        "group_col": None,
        "clean_texts": None,
        "tokens_list": None,
        "sentiment_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_processing():
    st.session_state.clean_texts = None
    st.session_state.tokens_list = None
    st.session_state.sentiment_df = None


def process_text(df: pd.DataFrame, text_col: str):
    clean_texts, tokens_list = preprocess_text(df[text_col])
    sentiment_df = get_sentiment(clean_texts)
    st.session_state.clean_texts = clean_texts
    st.session_state.tokens_list = tokens_list
    st.session_state.sentiment_df = sentiment_df


def main():
    st.title("AARC Survey Text Analytics Dashboard")
    st.caption("Explore survey responses with quick text analytics")

    init_state()
    if st.session_state.df is None or st.session_state.df.empty:
        df = load_data()
        if df.empty:
            st.error("Failed to load the bundled AARC survey data.")
            return
        st.session_state.df = df
        reset_processing()

    df = st.session_state.df
    st.success(f"Active dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns from the bundled AARC survey file.")

    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_columns:
        st.error("No text columns found in the dataset.")
        return

    with st.form("column_selection_form"):
        default_text_index = text_columns.index(st.session_state.text_col) if st.session_state.text_col in text_columns else 0
        text_col = st.selectbox("Select the text column for analysis", text_columns, index=default_text_index)
        group_col_options = [None] + list(df.columns)
        default_group_index = group_col_options.index(st.session_state.group_col) if st.session_state.group_col in group_col_options else 0
        group_col: Optional[str] = st.selectbox("Optional grouping column (for aggregation)", group_col_options, index=default_group_index)
        submitted = st.form_submit_button("Apply text settings")

    if submitted or st.session_state.clean_texts is None:
        st.session_state.text_col = text_col
        st.session_state.group_col = group_col
        process_text(df, text_col)
        st.success("Text preprocessing and sentiment analysis updated.")

    st.markdown(
        """
        ### How to use this app
        - The bundled `data/AARC-Survey-Responses.xlsx` file is preloaded automatically for all analyses.
        - Use the sidebar page navigation to open the overview, cleaning, sentiment, topic modeling, network, and export pages.
        - Return to this Home page anytime to switch the text column or grouping column used across the app.
        """
    )


if __name__ == "__main__":
    main()
