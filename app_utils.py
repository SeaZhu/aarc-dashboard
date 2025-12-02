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
        "text_col": "Response",
        "group_col": "Assigned.Category",
        "clean_texts": None,
        "tokens_list": None,
        "sentiment_df": None,
        "filter_choice": "All survey items",
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
        "overview": "📊 Overview",
        "textcleaningandngrams": "🧹 Text Cleaning & N-grams",
        "sentimentanalysis": "😊 Sentiment Analysis",
        "topicmodelinglda": "🧩 Topic Modeling (LDA)",
        "wordcooccurrencenetwork": "🕸️ Word Co-occurrence Network",
        "exportresults": "💾 Export Results",
    }
    st.markdown(
        f"""
        <script>
        const navLabels = {json.dumps(nav_label_overrides)};
        const normalize = (text) => text.toLowerCase().replace(/[^a-z0-9]/g, "");
        const updateNavLabels = () => {{
            const nav = window.parent.document.querySelector('[data-testid="stSidebarNav"]');
            if (!nav) return;
            const links = nav.querySelectorAll('a');
            links.forEach((link) => {{
                const labelEl =
                    link.querySelector('p') ||
                    link.querySelector('span') ||
                    link.querySelector('[data-testid="stSidebarNavLink"]');
                if (!labelEl) return;
                const current = labelEl.innerText.trim();
                const replacement = navLabels[normalize(current)];
                if (replacement && labelEl.innerText !== replacement) labelEl.textContent = replacement;
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
    """Load data, apply the survey-item filter, and process text from Response."""
    df = ensure_dataset_loaded()
    if df is None:
        return None

    filtered_df, filter_changed = render_text_settings_sidebar(df)
    if filtered_df is None or filtered_df.empty:
        st.warning("No responses available for the selected survey item.")
        return None

    if filter_changed or st.session_state.clean_texts is None:
        process_text(filtered_df)

    return filtered_df


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


def get_sentiment(raw_texts: pd.Series) -> pd.DataFrame:
    ensure_nltk()
    sia = SentimentIntensityAnalyzer()
    scores = []
    for text in raw_texts.fillna("").astype(str):
        score = sia.polarity_scores(text)
        sentiment = "neutral"
        if score["compound"] > 0.05:
            sentiment = "positive"
        elif score["compound"] < -0.05:
            sentiment = "negative"
        scores.append({"raw_text": text, **score, "sentiment": sentiment})
    return pd.DataFrame(scores)


def process_text(df: pd.DataFrame):
    text_col = "Response"
    clean_texts, tokens_list = preprocess_text(df[text_col])
    sentiment_df = get_sentiment(df[text_col])
    st.session_state.text_col = text_col
    st.session_state.group_col = "Assigned.Category"
    st.session_state.clean_texts = clean_texts
    st.session_state.tokens_list = tokens_list
    st.session_state.sentiment_df = sentiment_df


def render_text_settings_sidebar(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], bool]:
    """Render a single survey-item filter and return the filtered DataFrame."""
    unique_items = df["Survey.Item"].dropna().unique().tolist()
    options = ["All survey items"] + [f"Survey item {idx + 1}" for idx in range(len(unique_items))]
    default_index = options.index(st.session_state.filter_choice) if st.session_state.filter_choice in options else 0
    selection = st.sidebar.selectbox("Survey item filter", options, index=default_index)

    filter_changed = selection != st.session_state.filter_choice
    if filter_changed:
        st.session_state.filter_choice = selection
        reset_processing()

    if selection == "All survey items":
        return df, filter_changed

    try:
        selected_idx = options.index(selection) - 1
        target_item = unique_items[selected_idx]
    except (ValueError, IndexError):
        return df, filter_changed

    filtered = df[df["Survey.Item"] == target_item]
    return filtered, filter_changed


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


def topic_model(clean_texts: List[str], n_topics: int):
    vectorizer = CountVectorizer(max_features=1500)
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
    ax.set_title("Document Distribution Across Topics")
    st.pyplot(fig)
    st.caption("This chart shows how many responses were assigned to each topic by the model.")


def build_cooccurrence(tokens_list: List[List[str]], top_n: int = 20):
    all_tokens = list(itertools.chain.from_iterable(tokens_list))
    freq = Counter(all_tokens)
    top_terms = set([w for w, c in freq.most_common(top_n) if c > 1])
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
        st.warning("Text processing has not been completed. Use the sidebar survey-item filter to refresh the Response analytics.")
        return False
    if "tokens_list" not in st.session_state or st.session_state.tokens_list is None:
        st.warning("Token data missing. Use the sidebar survey-item filter to preprocess the Response column.")
        return False
    return True


def render_overview(df: pd.DataFrame):
    st.header("Dataset Overview")
    st.caption("This page provides an overview of the raw AARC survey comments.")

    st.markdown("**Dataset Preview (first 50 rows)**")
    st.dataframe(df.head(50))

    st.markdown("**Basic statistics**")
    summary_cols = ["Survey.Item", "Assigned.Category"]
    st.write(df[summary_cols].describe(include="all").transpose())

    st.markdown("**Sentiment Distribution by Category**")
    if st.session_state.sentiment_df is not None:
        combined = pd.concat([df.reset_index(drop=True), st.session_state.sentiment_df], axis=1)
        fig, ax = plt.subplots()
        sns.countplot(
            data=combined,
            x="Assigned.Category",
            hue="sentiment",
            palette={"positive": "#10b981", "neutral": "#d1d5db", "negative": "#ef4444"},
            ax=ax,
        )
        ax.set_xlabel("Assigned Category")
        ax.set_ylabel("Number of Responses")
        ax.legend(title="Sentiment")
        st.pyplot(fig)
    else:
        st.info("Sentiment results are unavailable. Please ensure preprocessing has completed.")


def render_cleaning(df: pd.DataFrame, text_col: str, clean_texts: List[str], tokens_list: List[List[str]]):
    st.header("Text Cleaning & N-grams")
    st.caption("Inspect cleaned survey responses and the most important words.")

    st.markdown("**Sample cleaned rows**")
    preview_df = pd.DataFrame({"original": df[text_col], "cleaned": clean_texts})
    preview_df = preview_df.drop_duplicates(subset=["cleaned"]).head(2)
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
        st.caption("The cloud highlights the most frequent tokens extracted from the Response text.")

    st.markdown("**Most frequent terms**")
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(data=freq_df.head(25), y="term", x="frequency", palette="mako", ax=ax)
    ax.set_title("Top Terms")
    st.pyplot(fig)


def render_sentiment(df: pd.DataFrame, text_col: str, group_col: Optional[str], sentiment_df: pd.DataFrame):
    st.header("Sentiment Analysis")
    st.caption("VADER-based polarity scores and distributions for each response.")

    combined = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
    st.markdown("**Per-response sentiment**")
    sentiment_table = combined[["Survey.Item", text_col, "sentiment", "compound"]]
    sentiment_table = sentiment_table.rename(columns={text_col: "Response"})
    st.dataframe(sentiment_table)

    st.markdown("**Sentiment distribution**")
    fig, ax = plt.subplots()
    order = ["negative", "neutral", "positive"]
    sns.countplot(
        data=combined,
        x="sentiment",
        order=order,
        palette={"positive": "#10b981", "neutral": "#d1d5db", "negative": "#ef4444"},
        ax=ax,
    )
    ax.set_xlabel("Sentiment label")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    if group_col and combined["compound"].nunique() > 1:
        st.markdown("**Average Sentiment by Category**")
        grouped = combined.groupby(group_col)["compound"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=grouped, x=group_col, y="compound", palette="viridis", ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.axhline(0, color="#9ca3af", linestyle="--", linewidth=1)
        st.pyplot(fig2)
    elif group_col:
        st.info("Average Sentiment by Category is hidden because all sentiment scores are identical.")


def render_topics(clean_texts: List[str]):
    st.header("Topic Modeling (LDA)")
    st.caption("Discover common themes across survey responses.")
    st.info(
        "Topics are built from cleaned Response text (lowercased, lemmatized, and stripped of stop words) "
        "to focus on the most informative tokens."
    )

    n_topics = st.slider("Number of topics", 2, 8, 3)
    try:
        topic_matrix, topic_keywords = topic_model(clean_texts, n_topics)
        topics_df = pd.DataFrame(topic_keywords)
        st.markdown("**Top keywords per topic**")
        st.dataframe(topics_df)

        if topic_matrix.shape[0] > 0 and len(set(topic_matrix.argmax(axis=1))) > 1:
            plot_topic_distribution(topic_matrix)
        else:
            st.info("Document distribution across topics is hidden because the assignments are repetitive.")
        return topics_df
    except ValueError:
        st.error("Not enough data to build topics. Try adjusting the topic count or cleaning options.")
        return None


def render_network(tokens_list: List[List[str]]):
    st.header("Word Co-occurrence Network")
    st.caption(
        "Relationships between high-frequency terms shown as a network graph. "
        "Co-occurrence counts reflect how often words appear together within the same response."
    )

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
