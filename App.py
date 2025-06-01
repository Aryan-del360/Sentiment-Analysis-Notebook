import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk # Ensure nltk is imported

# --- Configuration & Setup (MUST be the absolute first Streamlit command) ---
st.set_page_config(
    page_title="Product Sentiment Explorer 🛍️",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NLTK Data Download (Now defined/called AFTER st.set_page_config) ---
# Use st.cache_resource to download NLTK data only once per deployment/session
@st.cache_resource
def download_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        st.success("NLTK 'vader_lexicon' already downloaded and found. 🎉")
    except LookupError:
        st.info("Downloading NLTK 'vader_lexicon' (first run). This might take a moment... ⏳")
        nltk.download('vader_lexicon', quiet=True)
        st.success("NLTK 'vader_lexicon' downloaded successfully! ✅")
    return SentimentIntensityAnalyzer() # Return the initialized analyzer

# Call the function to ensure data is downloaded and analyzer is ready
analyzer = download_nltk_vader()

# --- Sentiment Analysis Functions ---
def get_textblob_sentiment(text):
    """Analyzes text sentiment using TextBlob."""
    if not isinstance(text, str):
        return 'Neutral' # Handle non-string input
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def get_vader_sentiment(text):
    """Analyzes text sentiment using NLTK VADER."""
    if not isinstance(text, str):
        return 'Neutral' # Handle non-string input
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.05:
        return 'Positive'
    elif vs['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# --- Data Loading and Analysis ---
@st.cache_data # Cache data to avoid reloading on every interaction
def load_and_analyze_data(file_path):
    """Loads CSV, performs sentiment analysis, and returns DataFrame."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Dataset '{file_path}' not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # --- IMPORTANT FIX: Convert 'Rate' column to numeric, coercing errors to NaN ---
    if 'Rate' in df.columns:
        df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
    # --- END OF FIX ---

    # --- IMPORTANT: Column names for text data and product name ---
    # The text column in Dataset-SA.csv is 'Review' as per your README.md
    if 'Review' not in df.columns:
        st.error("Error: 'Review' column not found in your CSV. Please check your dataset or update the column name in app.py if it's different.")
        st.stop()
    if 'product_name' not in df.columns:
        st.error("Error: 'product_name' column not found in your CSV. This is needed for product-wise analysis.")
        st.stop()

    # Apply sentiment analysis to the 'Review' column
    df['Sentiment_TextBlob'] = df['Review'].apply(get_textblob_sentiment)
    df['Sentiment_VADER'] = df['Review'].apply(get_vader_sentiment)

    # For a more detailed breakdown, could also store scores
    df['VADER_Compound'] = df['Review'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)

    return df

# Load the dataset
# Ensure 'Dataset-SA.csv' is in the same directory as this app.py
df = load_and_analyze_data('Dataset-SA.csv')

# --- Title & Introduction ---
st.title("🗣️ Product Review Sentiment Dashboard")
st.markdown("""
Welcome to your interactive hub for **understanding customer sentiment across different products!** 🚀
Use the filters to dive deep into review sentiment for specific products, explore trends, and gain actionable insights.
""")

# --- Sidebar Filters ---
st.sidebar.header("🎯 Filter Your View")

# Filter by Product Name
all_products = ['All Products'] + sorted(list(df['product_name'].unique()))
selected_product = st.sidebar.selectbox(
    "Select a Product:",
    options=all_products,
    index=0 # 'All Products' selected by default
)

# Filter by VADER Sentiment
vader_sentiments = ['All'] + list(df['Sentiment_VADER'].unique())
selected_vader_sentiment = st.sidebar.selectbox(
    "Filter by VADER Sentiment:",
    options=vader_sentiments,
    index=0
)

# Filter by TextBlob Sentiment
textblob_sentiments = ['All'] + list(df['Sentiment_TextBlob'].unique())
selected_textblob_sentiment = st.sidebar.selectbox(
    "Filter by TextBlob Sentiment:",
    options=textblob_sentiments,
    index=0
)

# Apply filters
filtered_df = df.copy()
if selected_product != 'All Products':
    filtered_df = filtered_df[filtered_df['product_name'] == selected_product]

if selected_vader_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment_VADER'] == selected_vader_sentiment]
if selected_textblob_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment_TextBlob'] == selected_textblob_sentiment]


# --- Main Dashboard Content ---
if filtered_df.empty:
    st.warning("😬 No data matches your current filter selections. Try adjusting them!")
else:
    st.markdown("---") # Visual separator
    st.header("📊 Filtered Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Reviews in Dataset", value=f"{len(df):,.0f} 📝")
    with col2:
        st.metric(label="Reviews Matching Filters", value=f"{len(filtered_df):,.0f} ✅")
    with col3:
        # Check if 'Rate' column exists before trying to calculate mean
        # The 'errors='coerce' in pd.to_numeric ensures non-numeric become NaN, which mean() can handle.
        if 'Rate' in filtered_df.columns and not filtered_df['Rate'].isnull().all():
            avg_overall_rating = filtered_df['Rate'].mean()
            st.metric(label="Avg. Product Rating", value=f"{avg_overall_rating:.1f} ⭐")
        else:
            st.metric(label="Avg. Product Rating", value="N/A")
    with col4:
        avg_compound_score = filtered_df['VADER_Compound'].mean()
        st.metric(label="Avg. VADER Compound Score", value=f"{avg_compound_score:.2f} 🌟")


    # --- Visualizations ---
    st.markdown("---")
    st.header("📈 Sentiment Distribution")

    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        st.subheader("VADER Sentiment Breakdown")
        vader_counts = filtered_df['Sentiment_VADER'].value_counts(normalize=True) * 100
        fig1, ax1 = plt.subplots(figsize=(7, 7))
        order = ['Positive', 'Neutral', 'Negative']
        colors = {'Positive': 'lightgreen', 'Neutral': 'skyblue', 'Negative': 'salmon'}
        vader_counts = vader_counts.reindex(order, fill_value=0)
        ax1.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90,
                colors=[colors[label] for label in vader_counts.index])
        ax1.set_title('VADER Sentiment Share')
        st.pyplot(fig1)

    with col_viz2:
        st.subheader("TextBlob Sentiment Breakdown")
        textblob_counts = filtered_df['Sentiment_TextBlob'].value_counts(normalize=True) * 100
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        textblob_counts = textblob_counts.reindex(order, fill_value=0)
        ax2.pie(textblob_counts, labels=textblob_counts.index, autopct='%1.1f%%', startangle=90,
                colors=[colors[label] for label in textblob_counts.index])
        ax2.set_title('TextBlob Sentiment Share')
        st.pyplot(fig2)

    st.markdown("""
    **Key Takeaways from Sentiment Distribution:**
    * **Overall Mood:** Get a quick feel for the dominant sentiment within the selected product(s).
    * **Method Comparison:** Observe if VADER and TextBlob offer similar conclusions, or if there are differences.
    * **Actionable Insights:** A high percentage of negative sentiment might indicate product-specific issues needing immediate attention.
    """)

    # --- Product-wise Sentiment Breakdown (if 'All Products' selected) ---
    if selected_product == 'All Products' and len(filtered_df['product_name'].unique()) > 1:
        st.markdown("---")
        st.header("📦 Sentiment Breakdown Per Product (VADER)")

        # Calculate positive sentiment percentage per product
        product_sentiment_vader = filtered_df.groupby('product_name')['Sentiment_VADER'].value_counts(normalize=True).unstack(fill_value=0)
        product_positive_percentage = product_sentiment_vader.get('Positive', pd.Series(dtype=float)) * 100

        # Filter out products with 0 positive reviews or no data, then sort
        product_positive_percentage = product_positive_percentage[product_positive_percentage > 0].sort_values(ascending=False)

        if not product_positive_percentage.empty:
            fig_prod, ax_prod = plt.subplots(figsize=(12, min(7 + len(product_positive_percentage) * 0.5, 20))) # Adjust fig size dynamically
            sns.barplot(x=product_positive_percentage.values, y=product_positive_percentage.index, ax=ax_prod, palette='Greens_r')
            ax_prod.set_title('Percentage of Positive Reviews Per Product (VADER)')
            ax_prod.set_xlabel('% Positive Reviews')
            ax_prod.set_ylabel('Product Name')
            st.pyplot(fig_prod)

            st.markdown("""
            **Insights Per Product:**
            * **Top Performers:** Easily spot products with the highest percentage of positive reviews – these are your winners! 🎉
            * **Areas for Improvement:** Identify products with lower positive sentiment, signaling potential issues that marketing or product teams should investigate.
            * **Comparative View:** Understand how products stack up against each other in terms of customer satisfaction.
            """)
        else:
            st.info("No sufficient data to show product-wise sentiment breakdown for current filters.")


    # --- Individual Review Analysis ---
    st.markdown("---")
    st.header("✍️ Analyze Custom Text")
    user_text = st.text_area(
        "Paste any sentence or paragraph here to analyze its sentiment:",
        "This product is absolutely fantastic and exceeded all my expectations! Highly recommend for everyone."
    )

    if st.button("Get Sentiment Spark!"):
        if user_text:
            st.subheader("Your Text's Sentiment:")

            # TextBlob Analysis
            textblob_result = get_textblob_sentiment(user_text)

            # VADER Analysis
            vader_scores = analyzer.polarity_scores(user_text)
            vader_result = get_vader_sentiment(user_text)

            st.markdown(f"**Original Text:** \"{user_text}\"")
            st.markdown(f"**TextBlob Prediction:** <span style='background-color:#ADD8E6; padding: 5px 10px; border-radius: 5px;'>**{textblob_result}**</span>", unsafe_allow_html=True)
            st.markdown(f"**VADER Prediction:** <span style='background-color:#90EE90; padding: 5px 10px; border-radius: 5px;'>**{vader_result}**</span>", unsafe_allow_html=True)
            st.markdown(f"**VADER Polarity Scores:** (Negative: {vader_scores['neg']:.2f}, Neutral: {vader_scores['neu']:.2f}, Positive: {vader_scores['pos']:.2f}, Compound: {vader_scores['compound']:.2f})")

            st.markdown("""
            * **TextBlob Polarity:** Ranges from -1 (most negative) to +1 (most positive).
            * **VADER Compound Score:** A normalized, weighted composite score, typically between -1 (most extreme negative) and +1 (most extreme positive).
            """)
        else:
            st.warning("Please enter some text to analyze. Don't leave it blank!")

    st.markdown("---")
    st.info("💡 Pro Tip: Use the filters in the sidebar to narrow down your analysis to specific products or...