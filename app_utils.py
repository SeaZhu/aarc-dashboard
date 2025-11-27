import itertools
import json
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
def load_data() -> pd.DataFrame:
    """Load the bundled AARC survey responses from the local data folder."""
    return pd.read_excel("data/AARC-Survey-Responses.xlsx")


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


def hide_main_nav_entry():
    """Hide the implicit entry for the root script so only named pages appear."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] ul li:first-child {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav_label_overrides = {
        "Overview": "📊 Overview",
        "Text Cleaning and Ngrams": "🧹 Text Cleaning & N-grams",
        "Sentiment Analysis": "😊 Sentiment Analysis",
        "Topic Modeling Lda": "🧩 Topic Modeling (LDA)",
        "Word Cooccurrence Network": "🕸️ Word Co-occurrence Network",
        "Export Results": "💾 Export Results",
    }
    st.markdown(
        f"""
        <script>
        const navLabels = {json.dumps(nav_label_overrides)};
        const updateNavLabels = () => {{
            const nav = window.parent.document.querySelector('[data-testid="stSidebarNav"] ul');
            if (!nav) return;
            const links = nav.querySelectorAll('a');
            links.forEach((link) => {{
                const labelEl = link.querySelector('p');
                if (!labelEl) return;
                const current = labelEl.innerText.trim();
                const replacement = navLabels[current];
                if (replacement && labelEl.innerText !== replacement) {{
                    labelEl.innerText = replacement;
                }}
            }});
        }};

        const navContainer = window.parent.document.querySelector('[data-testid="stSidebarNav"]');
        if (navContainer) {{
            const observer = new MutationObserver(updateNavLabels);
            observer.observe(navContainer, {{ childList: true, subtree: true }});
        }}

        updateNavLabels();
        </script>
        """,
        unsafe_allow_html=True,
    )


def ensure_dataset_loaded() -> Optional[pd.DataFrame]:
    """Load and cache the bundled dataset into session state if needed."""
    init_state()
    if st.session_state.df is None or st.session_state.df.empty:
        df = load_data()
        if df.empty:
            st.error("Failed to load the bundled AARC survey data.")
            return None
        st.session_state.df = df
        reset_processing()
    return st.session_state.df


def ensure_data_with_sidebar() -> Optional[pd.DataFrame]:
    """Load data if needed and render the shared text settings sidebar."""
    df = ensure_dataset_loaded()
    if df is None:
        return None
    if not render_text_settings_sidebar(df):
        return None
    return df


def reset_processing():
    st.session_state.clean_texts = None
    st.session_state.tokens_list = None
    st.session_state.sentiment_df = None


def preprocess_text(series: pd.Series) -> Tuple[List[str], List[List[str]]]:
    ensure_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_texts: List[str] = []
    tokenized: List[List[str]] = []
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


def process_text(df: pd.DataFrame, text_col: str):
    clean_texts, tokens_list = preprocess_text(df[text_col])
    sentiment_df = get_sentiment(clean_texts)
    st.session_state.clean_texts = clean_texts
    st.session_state.tokens_list = tokens_list
    st.session_state.sentiment_df = sentiment_df


def render_text_settings_sidebar(df: pd.DataFrame) -> bool:
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_columns:
        st.sidebar.error("No text columns found in the dataset.")
        return False

    with st.sidebar.form("column_selection_form"):
        st.subheader("Text settings")
        st.caption("Adjust the text and optional grouping columns used across all pages.")
        default_text_index = text_columns.index(st.session_state.text_col) if st.session_state.text_col in text_columns else 0
        text_col = st.selectbox("Text column", text_columns, index=default_text_index)
        group_col_options = [None] + list(df.columns)
        default_group_index = (
            group_col_options.index(st.session_state.group_col) if st.session_state.group_col in group_col_options else 0
        )
        group_col: Optional[str] = st.selectbox("Grouping column (optional)", group_col_options, index=default_group_index)
        submitted = st.form_submit_button("Apply settings")

    if submitted or st.session_state.clean_texts is None:
        st.session_state.text_col = text_col
        st.session_state.group_col = group_col
        process_text(df, text_col)
        st.sidebar.success("Text preprocessing and sentiment analysis updated.")

    return True


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
        topic_keywords[f"Topic {idx + 1}"] = words[top_indices].tolist()
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
    graph = nx.Graph()
    for (a, b, weight) in edges:
        graph.add_edge(a, b, weight=weight)
    sizes = [300 + freq[node] * 20 for node in graph.nodes()]
    pos = nx.spring_layout(graph, k=0.6, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(
        graph, pos, width=[graph[u][v]["weight"] for u, v in graph.edges()], alpha=0.5, edge_color="#888"
    )
    nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color="#4c78a8", alpha=0.8)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="white")
    ax.axis("off")
    st.pyplot(fig)


def download_button(df: pd.DataFrame, label: str, filename: str):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(label, data=buffer.getvalue(), file_name=filename, mime="text/csv")


def require_processed_data() -> bool:
    df = ensure_dataset_loaded()
    if df is None:
        return False
    if "clean_texts" not in st.session_state or st.session_state.clean_texts is None:
        st.warning("Text processing has not been completed. Use the sidebar text settings to configure the text column.")
        return False
    if "tokens_list" not in st.session_state or st.session_state.tokens_list is None:
        st.warning("Token data missing. Use the sidebar text settings to preprocess the text column.")
        return False
    return True


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
    st.header("Text Cleaning & N-grams")
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


def render_sentiment(df: pd.DataFrame, text_col: str, group_col: Optional[str], sentiment_df: pd.DataFrame):
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
