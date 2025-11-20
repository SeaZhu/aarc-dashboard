import itertools
from collections import Counter
from io import BytesIO
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


st.set_page_config(
    page_title="Text Analytics App - AARC Survey",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


@st.cache_data(show_spinner=True)
def load_data(source: str, upload, url: str) -> pd.DataFrame:
    if source == "sample":
        path = "data/AARC-Survey-Responses.xlsx"
        return pd.read_excel(path)
    if source == "upload" and upload is not None:
        if upload.name.endswith(".csv"):
            return pd.read_csv(upload)
        return pd.read_excel(upload)
    if source == "url" and url:
        if url.lower().endswith(".csv"):
            return pd.read_csv(url)
        return pd.read_excel(url)
    return pd.DataFrame()


def preprocess_text(series: pd.Series) -> Tuple[List[str], List[List[str]]]:
    ensure_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_texts = []
    tokenized = []
    for raw in series.fillna("").astype(str):
        tokens = word_tokenize(raw.lower())
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [lemmatizer.lemmatize(t) if lemmatizer else stemmer.stem(t) for t in tokens]
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        tokenized.append(tokens)
        clean_texts.append(" ".join(tokens))
    return clean_texts, tokenized


def get_sentiment(clean_texts: List[str]) -> pd.DataFrame:
    ensure_nltk()
    sia = SentimentIntensityAnalyzer()
    scores = []
    for text in clean_texts:
        score = sia.polarity_scores(text)
        sentiment = "neutral"
        if score["compound"] > 0.05:
            sentiment = "positive"
        elif score["compound"] < -0.05:
            sentiment = "negative"
        scores.append({"text": text, **score, "sentiment": sentiment})
    return pd.DataFrame(scores)


def plot_wordcloud(frequencies: dict):
    if not frequencies:
        st.info("Not enough text to build a word cloud.")
        return
    wc = WordCloud(width=900, height=400, background_color="white")
    wc.generate_from_frequencies(frequencies)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def topic_model(clean_texts: List[str], n_topics: int, max_features: int = 1000):
    vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
    dtm = vectorizer.fit_transform(clean_texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topics = lda.fit_transform(dtm)
    words = np.array(vectorizer.get_feature_names_out())
    topic_keywords = {}
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-11:-1]
        topic_keywords[f"Topic {idx+1}"] = words[top_indices].tolist()
    return topics, topic_keywords


def plot_topic_distribution(topic_matrix):
    topic_counts = topic_matrix.argmax(axis=1)
    fig, ax = plt.subplots()
    sns.countplot(x=topic_counts + 1, palette="viridis", ax=ax)
    ax.set_xlabel("Most Likely Topic")
    ax.set_ylabel("Number of Responses")
    st.pyplot(fig)


def build_cooccurrence(tokens_list: List[List[str]], top_n: int = 20):
    all_tokens = list(itertools.chain.from_iterable(tokens_list))
    freq = Counter(all_tokens)
    top_terms = set([w for w, _ in freq.most_common(top_n)])
    co_counts = Counter()
    for tokens in tokens_list:
        unique_tokens = [t for t in set(tokens) if t in top_terms]
        for a, b in itertools.combinations(sorted(unique_tokens), 2):
            co_counts[(a, b)] += 1
    return freq, co_counts


def plot_network(freq: Counter, co_counts: Counter, min_co: int = 2):
    edges = [(a, b, w) for (a, b), w in co_counts.items() if w >= min_co]
    if not edges:
        st.info("Not enough co-occurrence data to build a network.")
        return
    G = nx.Graph()
    for (a, b, w) in edges:
        G.add_edge(a, b, weight=w)
    sizes = []
    for node in G.nodes():
        sizes.append(300 + freq[node] * 20)
    pos = nx.spring_layout(G, k=0.6, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(
        G, pos, width=[G[u][v]["weight"] for u, v in G.edges()], alpha=0.5, edge_color="#888"
    )
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4c78a8", alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white")
    ax.axis("off")
    st.pyplot(fig)


def download_button(df: pd.DataFrame, label: str, filename: str):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(label, data=buffer.getvalue(), file_name=filename, mime="text/csv")


def render_overview(df: pd.DataFrame):
    st.header("Dataset Overview")
    st.caption("Preview and explore the structure of the loaded survey responses.")

    st.markdown("**Preview (first 50 rows)**")
    st.dataframe(df.head(50))

    st.markdown("**Basic statistics**")
    st.write(df.describe(include="all").transpose())

    st.markdown("**Value counts by column**")
    col1, col2 = st.columns(2)
    with col1:
        count_col = st.selectbox("Select column for counts", df.columns)
    with col2:
        top_n = st.slider("Show top N values", 5, 30, 10, key="overview_topn")
    counts = df[count_col].value_counts().head(top_n)
    st.bar_chart(counts)


def render_cleaning(df: pd.DataFrame, text_col: str, clean_texts: List[str], tokens_list: List[List[str]]):
    st.header("Text Cleaning & Keywords")
    st.caption("Inspect cleaned survey responses and the most important words.")

    st.markdown("**Sample cleaned rows**")
    preview_df = pd.DataFrame({"original": df[text_col].head(10), "cleaned": clean_texts[:10]})
    st.dataframe(preview_df)

    frequencies = Counter(itertools.chain.from_iterable(tokens_list))
    freq_df = pd.DataFrame(frequencies.most_common(50), columns=["term", "frequency"])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Top keywords**")
        st.dataframe(freq_df)
    with col2:
        st.markdown("**Word Cloud**")
        plot_wordcloud(dict(frequencies))

    st.markdown("**Most frequent terms**")
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(data=freq_df.head(25), y="term", x="frequency", palette="mako", ax=ax)
    ax.set_title("Top Terms")
    st.pyplot(fig)


def render_sentiment(df: pd.DataFrame, text_col: str, group_col: str, sentiment_df: pd.DataFrame):
    st.header("Sentiment Analysis")
    st.caption("VADER-based polarity scores and distributions for each response.")

    combined = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    st.markdown("**Per-response sentiment**")
    st.dataframe(combined[[text_col, "compound", "sentiment"]])

    st.markdown("**Sentiment distribution**")
    fig, ax = plt.subplots()
    sns.countplot(data=combined, x="sentiment", palette=["#ef4444", "#d1d5db", "#10b981"], ax=ax)
    ax.set_xlabel("Sentiment label")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    if group_col:
        st.markdown("**Average sentiment by group**")
        grouped = combined.groupby(group_col)["compound"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=grouped, x=group_col, y="compound", palette="viridis", ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.axhline(0, color="#9ca3af", linestyle="--", linewidth=1)
        st.pyplot(fig2)


def render_topics(clean_texts: List[str]):
    st.header("Topic Modeling (LDA)")
    st.caption("Discover common themes across survey responses.")

    n_topics = st.slider("Number of topics", 2, 8, 3)
    max_features = st.slider("Max vocabulary size", 300, 2000, 1000, step=100)
    try:
        topic_matrix, topic_keywords = topic_model(clean_texts, n_topics, max_features=max_features)
        topics_df = pd.DataFrame(topic_keywords)
        st.markdown("**Top keywords per topic**")
        st.dataframe(topics_df)
        plot_topic_distribution(topic_matrix)
        return topics_df
    except ValueError:
        st.error("Not enough data to build topics. Try adjusting the topic count or cleaning options.")
        return None


def render_network(tokens_list: List[List[str]]):
    st.header("Word Co-occurrence Network")
    st.caption("Relationships between high-frequency terms shown as a network graph.")

    top_n_terms = st.slider("Top N terms to consider", 10, 50, 20)
    min_co = st.slider("Minimum co-occurrence to display", 1, 5, 2)
    freq, co_counts = build_cooccurrence(tokens_list, top_n_terms)
    plot_network(freq, co_counts, min_co)


def render_export(df: pd.DataFrame, clean_texts: List[str], sentiment_df: pd.DataFrame, topics_df: Optional[pd.DataFrame]):
    st.header("Export Analysis")
    st.caption("Download cleaned text, sentiment, or topic outputs as CSV files.")

    combined = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    download_button(combined, "Download sentiment results", "sentiment_results.csv")
    if topics_df is not None:
        download_button(topics_df, "Download topic keywords", "topic_keywords.csv")
    cleaned_df = pd.DataFrame({"cleaned_text": clean_texts})
    download_button(cleaned_df, "Download cleaned text", "cleaned_text.csv")


def main():
    st.title("AARC Survey Text Analytics Dashboard")
    st.caption("Explore survey responses with quick text analytics")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        [
            "Dataset Overview",
            "Text Cleaning & Keywords",
            "Sentiment Analysis",
            "Topic Modeling",
            "Co-occurrence Network",
            "Export Results",
        ],
    )

    st.sidebar.header("Data Source")
    source = st.sidebar.radio(
        "Choose data source", ["sample", "upload", "url"], index=0, format_func=lambda x: {
            "sample": "Use bundled sample",
            "upload": "Upload CSV/XLSX",
            "url": "GitHub raw URL",
        }[x],
    )
    upload = st.sidebar.file_uploader("Upload survey file", type=["csv", "xlsx"])
    url = st.sidebar.text_input("GitHub raw URL (CSV/XLSX)")

    df = load_data(source, upload, url)
    if df.empty:
        st.warning("Load a dataset to get started.")
        return

    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_columns:
        st.error("No text columns found in the dataset.")
        return

    text_col = st.selectbox("Select the text column for analysis", text_columns)
    group_col = st.selectbox("Optional grouping column (for aggregation)", [None] + list(df.columns))

    clean_texts, tokens_list = preprocess_text(df[text_col])
    sentiment_df = get_sentiment(clean_texts)

    topics_df: Optional[pd.DataFrame] = None
    if page == "Dataset Overview":
        render_overview(df)
    elif page == "Text Cleaning & Keywords":
        render_cleaning(df, text_col, clean_texts, tokens_list)
    elif page == "Sentiment Analysis":
        render_sentiment(df, text_col, group_col, sentiment_df)
    elif page == "Topic Modeling":
        topics_df = render_topics(clean_texts)
    elif page == "Co-occurrence Network":
        render_network(tokens_list)
    elif page == "Export Results":
        export_topics = st.slider("Topics to include in export", 2, 8, 3)
        export_features = st.slider("Max vocabulary for export topics", 300, 2000, 1000, step=100)
        try:
            _, topic_keywords = topic_model(clean_texts, export_topics, max_features=export_features)
            topics_df = pd.DataFrame(topic_keywords)
        except ValueError:
            topics_df = None
        render_export(df, clean_texts, sentiment_df, topics_df)


if __name__ == "__main__":
    main()
