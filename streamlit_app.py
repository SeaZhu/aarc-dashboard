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

    st.sidebar.header("Data Source")
    with st.sidebar.form("data_source_form"):
        source = st.radio(
            "Choose data source",
            ["sample", "upload", "url"],
            index=0,
            format_func=lambda x: {
                "sample": "Use bundled sample",
                "upload": "Upload CSV/XLSX",
                "url": "GitHub raw URL",
            }[x],
        )
        upload = st.file_uploader("Upload survey file", type=["csv", "xlsx"], key="upload_widget")
        url = st.text_input("GitHub raw URL (CSV/XLSX)")
        load_clicked = st.form_submit_button("Load data")

    if load_clicked:
        df = load_data(source, upload, url)
        if df.empty:
            st.error("No data loaded. Please verify the file or URL.")
        else:
            st.session_state.df = df
            reset_processing()
            st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    df = st.session_state.df
    if df is None or df.empty:
        st.info("Use the sidebar to load a dataset. All feature pages will become available once data is loaded.")
        return

    st.success(f"Active dataset: {df.shape[0]} rows and {df.shape[1]} columns.")

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
        - Use the built-in Streamlit sidebar navigation to open pages for overview, cleaning, sentiment, topics, network, and export.
        - Return to this Home page anytime to switch datasets or change the text column used across all analyses.
        - Each page will automatically consume the processed text, tokens, and sentiment stored in the session state.
        """
    )


if __name__ == "__main__":
    main()
