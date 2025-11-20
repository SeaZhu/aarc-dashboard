import streamlit as st

from app_utils import render_network, require_processed_data


st.set_page_config(page_title="Co-occurrence Network", page_icon="🕸️")


def main():
    st.title("Word Co-occurrence Network")
    if not require_processed_data():
        return

    render_network(st.session_state.tokens_list)


if __name__ == "__main__":
    main()
