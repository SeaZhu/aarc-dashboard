import streamlit as st

from app_utils import ensure_data_with_sidebar, hide_main_nav_entry, render_overview


st.set_page_config(
    page_title="Text Analytics App - Overview",
    page_icon="🧠",
    layout="wide",
)


def main():
    hide_main_nav_entry()
    try:
        st.switch_page("pages/01_Overview.py")
        return
    except Exception:
        pass

    st.title("Dataset Overview")
    st.caption("The bundled AARC survey data is preloaded for you.")

    df = ensure_data_with_sidebar()
    if df is None or df.empty:
        return

    st.success(
        f"Active dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns from the bundled AARC survey file."
    )

    render_overview(df)


if __name__ == "__main__":
    main()
