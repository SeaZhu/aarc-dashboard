import streamlit as st

from app_utils import ensure_data_with_sidebar, hide_main_nav_entry, render_overview


st.set_page_config(page_title="Overview", page_icon="📊")


def main():
    hide_main_nav_entry()
    st.title("Overview")
    st.caption("Preview the bundled survey data and adjust text settings in the sidebar.")

    df = ensure_data_with_sidebar()
    if df is None or df.empty:
        return

    render_overview(df)


if __name__ == "__main__":
    main()
