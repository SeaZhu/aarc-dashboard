import streamlit as st

from app_utils import (
    init_state,
    load_data,
    render_overview,
    render_text_settings_sidebar,
    reset_processing,
)


st.set_page_config(page_title="01 Overview", page_icon="📊")


def prepare_data():
    init_state()
    if st.session_state.df is None or st.session_state.df.empty:
        df = load_data()
        if df.empty:
            st.error("Failed to load the bundled AARC survey data.")
            return None
        st.session_state.df = df
        reset_processing()
    return st.session_state.df


def main():
    st.title("Overview")
    st.caption("Preview the bundled survey data and adjust text settings in the sidebar.")

    df = prepare_data()
    if df is None or df.empty:
        return

    if not render_text_settings_sidebar(df):
        return

    render_overview(df)


if __name__ == "__main__":
    main()
