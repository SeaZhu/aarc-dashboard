import streamlit as st

st.set_page_config(
    page_title="Text Analytics App",
    page_icon="🧠",
    layout="wide",
)


def main():
    navigation = st.navigation(
        {
            "": [
                st.Page(page="pages/01_Overview.py", title="Overview", icon="📊"),
                st.Page(page="pages/02_Text_Cleaning_and_Ngrams.py", title="Text Cleaning & N-grams", icon="🧹"),
                st.Page(page="pages/03_Sentiment_Analysis.py", title="Sentiment Analysis", icon="😊"),
                st.Page(page="pages/04_Topic_Modeling_LDA.py", title="Topic Modeling (LDA)", icon="🧩"),
                st.Page(page="pages/05_Word_Cooccurrence_Network.py", title="Word Co-occurrence Network", icon="🕸️"),
                st.Page(page="pages/06_Export_Results.py", title="Export Results", icon="💾"),
            ]
        }
    )
    navigation.run()


if __name__ == "__main__":
    main()
