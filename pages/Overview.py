import streamlit as st

from app_utils import render_overview, require_processed_data


st.set_page_config(page_title="Dataset Overview", page_icon="📊")


def main():
    st.title("Dataset Overview")
    if not require_processed_data():
        return

    df = st.session_state.df
    render_overview(df)


if __name__ == "__main__":
    main()
