import pandas as pd
import streamlit as st

from app_utils import render_export, require_processed_data, topic_model


st.set_page_config(page_title="06 Export Results", page_icon="💾")


def main():
    st.title("Export Results")
    if not require_processed_data():
        return

    clean_texts = st.session_state.clean_texts
    df = st.session_state.df
    sentiment_df = st.session_state.sentiment_df

    st.markdown("Configure topic regeneration for export (optional)")
    topics_df = None
    with st.form("export_topic_form"):
        export_topics = st.slider("Topics to include in export", 2, 8, 3)
        export_features = st.slider("Max vocabulary for export topics", 300, 2000, 1000, step=100)
        regenerate = st.form_submit_button("Regenerate topic keywords")

    if regenerate:
        try:
            _, topic_keywords = topic_model(clean_texts, export_topics, max_features=export_features)
            topics_df = pd.DataFrame(topic_keywords)
            st.success("Topic keywords regenerated for export.")
        except ValueError:
            st.warning("Not enough data to build topics for export. Try different settings.")

    render_export(df, clean_texts, sentiment_df, topics_df)


if __name__ == "__main__":
    main()
