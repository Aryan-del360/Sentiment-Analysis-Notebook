import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- NLTK Data Download (Crucial for Deployment) ---
# Ensure VADER lexicon is available. In Dockerfile, you already have:
# RUN python -m nltk.downloader vader_lexicon
# For local testing, you might need to uncomment and run this once:
# try:
#     nltk.data.find('sentiment/vader_lexicon.zip')
# except nltk.downloader.DownloadError:
#     nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Sentiment Spark ✨",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up matplotlib style for a cleaner look
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

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

    # --- IMPORTANT CORRECTION ---
    # The text column in Dataset-SA.csv is 'Review' as per your README.md
    if 'Review' not in df.columns:
        st.error("Error: 'Review' column not found in your CSV. Please check your dataset or update the column name in app.py if it's different.")
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
st.title("🗣️ Sentiment Pulse Dashboard")
st.markdown("""
Welcome to your go-to dashboard for **unlocking insights from text data!** ✨
Quickly grasp the overall sentiment distribution from your dataset and dive into individual text analysis.
""")

# --- Sidebar Filters ---
st.sidebar.header("🎯 Filter Options")

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
if selected_vader_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment_VADER'] == selected_vader_sentiment]
if selected_textblob_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment_TextBlob'] == selected_textblob_sentiment]

# --- Main Dashboard Content ---
if filtered_df.empty:
    st.warning("😬 No data matches your current filter selections. Try adjusting them!")
else:
    st.markdown("---") # Visual separator
    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Reviews Analyzed", value=f"{len(df):,.0f} 📝")
    with col2:
        st.metric(label="Reviews Matching Filters", value=f"{len(filtered_df):,.0f} ✅")
    with col3:
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
        # Ensure order and colors for consistency
        order = ['Positive', 'Neutral', 'Negative']
        colors = {'Positive': 'lightgreen', 'Neutral': 'skyblue', 'Negative': 'salmon'}
        
        # Plot only categories that exist in the filtered data
        vader_counts = vader_counts.reindex(order, fill_value=0)
        
        ax1.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90,
                colors=[colors[label] for label in vader_counts.index])
        ax1.set_title('VADER Sentiment Share')
        st.pyplot(fig1)

    with col_viz2:
        st.subheader("TextBlob Sentiment Breakdown")
        textblob_counts = filtered_df['Sentiment_TextBlob'].value_counts(normalize=True) * 100
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        
        # Ensure order and colors for consistency
        textblob_counts = textblob_counts.reindex(order, fill_value=0)
        
        ax2.pie(textblob_counts, labels=textblob_counts.index, autopct='%1.1f%%', startangle=90,
                colors=[colors[label] for label in textblob_counts.index])
        ax2.set_title('TextBlob Sentiment Share')
        st.pyplot(fig2)
    
    st.markdown("""
    **Key Takeaways from Sentiment Distribution:**
    * **Overall Mood:** Get a quick feel for the dominant sentiment in your dataset (e.g., predominantly positive, or a balanced mix).
    * **Method Comparison:** Observe if VADER and TextBlob offer similar conclusions, or if there are notable differences in their classification.
    * **Actionable Insights:** A high percentage of negative sentiment might indicate areas needing immediate attention (e.g., product issues, service complaints).
    """)

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
    st.info("💡 Pro Tip: Filter the dataset using the sidebar to explore specific sentiment groups!")

    # Optional: Display Raw Data Table
    if st.checkbox("Show Raw Data Table (Filtered)"):
        st.subheader("Filtered Dataset Preview")
        st.dataframe(filtered_df)
