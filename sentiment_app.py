import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Optional: Download VADER lexicon if not already done
# import nltk
# try:
#     nltk.data.find('sentiment/vader_lexicon.zip')
# except nltk.downloader.DownloadError:
#     nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# --- Functions for Sentiment Analysis ---
def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def get_vader_sentiment(text):
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.05:
        return 'Positive'
    elif vs['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wide layout for better space

st.title("E-commerce Product Review Sentiment Analysis Dashboard")

st.markdown("""
This dashboard provides insights into sentiment expressed in product reviews.
You can explore overall sentiment trends or analyze individual reviews.
""")

# --- Data Loading (Example - replace with your actual data load) ---
@st.cache_data # Cache data to avoid reloading on every interaction
def load_data():
    # Replace with your actual data loading, e.g., from CSV, database
    data = {
        'review_text': [
            "This product is amazing! Highly recommend.",
            "Absolutely terrible, wasted my money.",
            "It's okay, nothing special.",
            "Works as expected, fairly good quality.",
            "Very disappointed with the purchase. Broke in a week.",
            "Excellent value for money and fast delivery."
        ],
        'product_id': [1, 2, 1, 3, 2, 3],
        'timestamp': pd.to_datetime(['2023-01-10', '2023-01-15', '2023-02-01', '2023-02-05', '2023-03-01', '2023-03-10'])
    }
    df = pd.DataFrame(data)
    # Apply sentiment analysis
    df['sentiment_textblob'] = df['review_text'].apply(get_textblob_sentiment)
    df['sentiment_vader'] = df['review_text'].apply(get_vader_sentiment)
    return df

df = load_data()

# --- Sidebar for Navigation/Filters ---
st.sidebar.header("Dashboard Options")
analysis_type = st.sidebar.radio(
    "Choose Analysis Type:",
    ('Overall Trends', 'Individual Review Analysis')
)

if analysis_type == 'Overall Trends':
    st.header("Overall Sentiment Trends")

    # Sentiment Distribution Chart
    st.subheader("Sentiment Distribution (VADER)")
    sentiment_counts = df['sentiment_vader'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', ax=ax1, color=['lightgreen', 'salmon', 'skyblue'])
    ax1.set_title('Distribution of Sentiments')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Number of Reviews')
    ax1.tick_params(axis='x', rotation=0) # Ensure labels are horizontal
    st.pyplot(fig1)

    # Sentiment over Time (if timestamp exists)
    if 'timestamp' in df.columns:
        st.subheader("Sentiment Trend Over Time")
        # For simplicity, let's just plot positive/negative counts
        df_time = df.groupby(df['timestamp'].dt.to_period('M'))['sentiment_vader'].value_counts().unstack(fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        df_time[['Positive', 'Negative']].plot(kind='line', marker='o', ax=ax2)
        ax2.set_title('Positive and Negative Reviews Over Time')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Reviews')
        ax2.legend(title='Sentiment')
        st.pyplot(fig2)

    # Optional: Display raw data
    if st.checkbox('Show Raw Data'):
        st.subheader("Raw Review Data")
        st.dataframe(df)

elif analysis_type == 'Individual Review Analysis':
    st.header("Analyze Individual Review")
    user_review = st.text_area("Enter a product review:", "This product is fantastic and exceeded my expectations!")

    if st.button("Analyze Sentiment"):
        if user_review:
            st.subheader("Analysis Results:")
            textblob_result = get_textblob_sentiment(user_review)
            vader_scores = analyzer.polarity_scores(user_review)
            vader_result = get_vader_sentiment(user_review)

            st.write(f"**Review:** \"{user_review}\"")
            st.write(f"**TextBlob Sentiment:** {textblob_result}")
            st.write(f"**VADER Sentiment:** {vader_result}")
            st.write(f"**VADER Polarity Scores:** (Negative: {vader_scores['neg']:.2f}, Neutral: {vader_scores['neu']:.2f}, Positive: {vader_scores['pos']:.2f}, Compound: {vader_scores['compound']:.2f})")
        else:
            st.warning("Please enter a review to analyze.")