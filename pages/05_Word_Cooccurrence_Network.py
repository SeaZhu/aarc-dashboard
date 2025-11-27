import streamlit as st

from app_utils import ensure_data_with_sidebar, render_network, require_processed_data


st.set_page_config(page_title="Word Co-occurrence Network", page_icon="🕸️")


def main():
    st.title("Word Co-occurrence Network")
    df = ensure_data_with_sidebar()
    if df is None or df.empty:
        return
    if not require_processed_data():
        return
    render_network(st.session_state.tokens_list)


if __name__ == "__main__":
    main()
