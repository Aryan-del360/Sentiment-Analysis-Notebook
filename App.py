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
    page_title="Product Sentiment Dashboard 🛍️",
    page_icon="📈",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded"
)

# --- NLTK Data Download (Cached to run once per deployment) ---
@st.cache_resource
def download_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        # st.success("NLTK 'vader_lexicon' already downloaded and found. 🎉") # Remove this from final app, it clutters
    except LookupError:
        st.info("Downloading NLTK 'vader_lexicon' (first run). This might take a moment... ⏳")
        nltk.download('vader_lexicon', quiet=True)
        st.success("NLTK 'vader_lexicon' downloaded successfully! ✅")
    return SentimentIntensityAnalyzer() # Return the initialized analyzer

# Call the function to ensure data is downloaded and analyzer is ready
analyzer = download_nltk_vader()

# --- Sentiment Analysis Functions (unchanged) ---
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

# --- Data Loading and Analysis (Cached for performance) ---
@st.cache_data
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

    # Convert 'Rate' column to numeric, coercing errors to NaN
    if 'Rate' in df.columns:
        df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
    else:
        st.warning("Column 'Rate' not found in dataset. Some metrics will be unavailable.")

    # Ensure required text columns exist
    if 'Review' not in df.columns:
        st.error("Error: 'Review' column not found in your CSV. Please check your dataset or update the column name in app.py if it's different.")
        st.stop()
    if 'product_name' not in df.columns:
        st.error("Error: 'product_name' column not found in your CSV. This is needed for product-wise analysis.")
        st.stop()

    # Apply sentiment analysis
    df['Sentiment_TextBlob'] = df['Review'].apply(get_textblob_sentiment)
    df['Sentiment_VADER'] = df['Review'].apply(get_vader_sentiment)
    df['VADER_Compound'] = df['Review'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)

    return df

# Load the dataset
df = load_and_analyze_data('Dataset-SA.csv')

# --- Main Title and Introduction ---
st.title("🛍️ E-commerce Product Review Sentiment Dashboard")
st.markdown("""
Welcome to your interactive dashboard for **understanding customer sentiment!**
Use the filters in the sidebar to refine your analysis, explore sentiment distributions,
and analyze individual reviews.
""")

# --- Sidebar for Filters ---
st.sidebar.header("🎯 Filter Options")

with st.sidebar.expander("Filter by Product"):
    all_products = ['All Products'] + sorted(list(df['product_name'].unique()))
    selected_product = st.selectbox(
        "Select a Product:",
        options=all_products,
        index=0 # 'All Products' by default
    )

with st.sidebar.expander("Filter by Sentiment Type"):
    st.markdown("**VADER Sentiment:**")
    vader_sentiments = ['All'] + list(df['Sentiment_VADER'].unique())
    selected_vader_sentiment = st.selectbox(
        "Filter by VADER Sentiment:",
        options=vader_sentiments,
        index=0,
        key="vader_filter"
    )
    st.markdown("**TextBlob Sentiment:**")
    textblob_sentiments = ['All'] + list(df['Sentiment_TextBlob'].unique())
    selected_textblob_sentiment = st.selectbox(
        "Filter by TextBlob Sentiment:",
        options=textblob_sentiments,
        index=0,
        key="textblob_filter"
    )

# Apply filters
filtered_df = df.copy()
if selected_product != 'All Products':
    filtered_df = filtered_df[filtered_df['product_name'] == selected_product]
if selected_vader_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment_VADER'] == selected_vader_sentiment]
if selected_textblob_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment_TextBlob'] == selected_textblob_sentiment]

# --- Main Content Area with Tabs ---
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Metrics", "📈 Sentiment Distribution", "📦 Product Insights", "✍️ Analyze Custom Text"])

with tab1:
    st.header("Overall Dashboard Metrics")
    if filtered_df.empty:
        st.warning("😬 No data matches your current filter selections. Try adjusting them!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Reviews (Dataset)", value=f"{len(df):,.0f} 📝")
        with col2:
            st.metric(label="Reviews Matching Filters", value=f"{len(filtered_df):,.0f} ✅")
        with col3:
            if 'Rate' in filtered_df.columns and not filtered_df['Rate'].isnull().all():
                avg_overall_rating = filtered_df['Rate'].mean()
                st.metric(label="Avg. Product Rating", value=f"{avg_overall_rating:.1f} ⭐")
            else:
                st.metric(label="Avg. Product Rating", value="N/A")
        with col4:
            avg_compound_score = filtered_df['VADER_Compound'].mean()
            st.metric(label="Avg. VADER Compound Score", value=f"{avg_compound_score:.2f} 🌟")

        st.markdown("---")
        st.subheader("Random Reviews Sample")
        st.dataframe(filtered_df[['product_name', 'Review', 'Rate', 'Sentiment_VADER', 'Sentiment_TextBlob']].sample(min(5, len(filtered_df))), use_container_width=True)

with tab2:
    st.header("Sentiment Distribution Visualizations")
    if filtered_df.empty:
        st.warning("😬 No data available for plots. Adjust filters!")
    else:
        col_viz1, col_viz2 = st.columns(2)
        order = ['Positive', 'Neutral', 'Negative']
        colors = {'Positive': '#66c2a5', 'Neutral': '#8da0cb', 'Negative': '#fc8d62'} # Softer, distinct colors

        with col_viz1:
            st.subheader("VADER Sentiment Breakdown")
            vader_counts = filtered_df['Sentiment_VADER'].value_counts(normalize=True) * 100
            fig1, ax1 = plt.subplots(figsize=(6, 6)) # Slightly smaller for tabs
            vader_counts = vader_counts.reindex(order, fill_value=0)
            ax1.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90,
                    colors=[colors[label] for label in vader_counts.index],
                    pctdistance=0.85, wedgeprops=dict(width=0.4)) # Donut chart
            centre_circle = plt.Circle((0,0),0.70,fc='white') # For donut chart
            fig1.gca().add_artist(centre_circle)
            ax1.set_title('VADER Sentiment Share', fontsize=14)
            st.pyplot(fig1)

        with col_viz2:
            st.subheader("TextBlob Sentiment Breakdown")
            textblob_counts = filtered_df['Sentiment_TextBlob'].value_counts(normalize=True) * 100
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            textblob_counts = textblob_counts.reindex(order, fill_value=0)
            ax2.pie(textblob_counts, labels=textblob_counts.index, autopct='%1.1f%%', startangle=90,
                    colors=[colors[label] for label in textblob_counts.index],
                    pctdistance=0.85, wedgeprops=dict(width=0.4)) # Donut chart
            centre_circle = plt.Circle((0,0),0.70,fc='white') # For donut chart
            fig2.gca().add_artist(centre_circle)
            ax2.set_title('TextBlob Sentiment Share', fontsize=14)
            st.pyplot(fig2)

        st.markdown("""
        * **VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
        * **TextBlob** is a Python library for processing textual data. It provides a simple API for diving into common NLP tasks like sentiment analysis.
        """)

with tab3:
    st.header("Product-wise Sentiment Insights")
    if selected_product != 'All Products':
        st.info(f"💡 You have selected a specific product (**{selected_product}**). This tab displays insights for 'All Products' when no specific product is filtered.")
    elif filtered_df.empty:
        st.warning("😬 No data available to show product insights. Adjust filters!")
    else:
        st.subheader("Percentage of Positive Reviews Per Product (VADER)")
        product_sentiment_vader = filtered_df.groupby('product_name')['Sentiment_VADER'].value_counts(normalize=True).unstack(fill_value=0)
        product_positive_percentage = product_sentiment_vader.get('Positive', pd.Series(dtype=float)) * 100

        # Filter out products with 0 positive reviews for clearer visualization
        product_positive_percentage = product_positive_percentage[product_positive_percentage > 0].sort_values(ascending=False)

        if not product_positive_percentage.empty:
            fig_prod, ax_prod = plt.subplots(figsize=(10, max(6, len(product_positive_percentage) * 0.4))) # Dynamic height
            sns.barplot(x=product_positive_percentage.values, y=product_positive_percentage.index, ax=ax_prod, palette='viridis')
            ax_prod.set_title('Top Products by Positive Sentiment (VADER)', fontsize=14)
            ax_prod.set_xlabel('% Positive Reviews')
            ax_prod.set_ylabel('Product Name')
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            st.pyplot(fig_prod)
            st.markdown("""
            This chart helps you quickly identify top-performing products based on the percentage of positive reviews.
            Products with lower percentages here might need further investigation.
            """)
        else:
            st.info("Not enough data to show product-wise sentiment breakdown for current filters.")

with tab4:
    st.header("Analyze Any Text for Sentiment")
    st.markdown("Want to test how the sentiment models work? Enter any text below!")
    user_text = st.text_area(
        "Paste your text here:",
        "This product is absolutely fantastic and exceeded all my expectations! Highly recommend for everyone.",
        height=150
    )

    if st.button("Get Sentiment Analysis!"):
        if user_text:
            st.subheader("Analysis Results:")
            textblob_result = get_textblob_sentiment(user_text)
            vader_scores = analyzer.polarity_scores(user_text)
            vader_result = get_vader_sentiment(user_text)

            st.markdown(f"**Original Text:** \"{user_text}\"")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label="TextBlob Prediction", value=textblob_result)
            with col_res2:
                st.metric(label="VADER Prediction", value=vader_result)

            st.markdown(f"**VADER Polarity Scores:**")
            st.json(vader_scores) # Display VADER scores in a nice JSON format
            st.markdown("""
            * **VADER Scores Explained:**
                * `neg`, `neu`, `pos`: Represent the proportion of negative, neutral, and positive words in the text (sum to 1.0).
                * `compound`: A normalized, weighted composite score which ranges from -1 (most extreme negative) to +1 (most extreme positive).
            """)
        else:
            st.warning("Please enter some text to analyze. The field cannot be empty!")

st.markdown("---")
with st.expander("📚 Data Dictionary & Usage Tips"):
    st.markdown("""
    **Dataset Columns:**
    * `product_name`: Name of the reviewed product.
    * `product_price`: Price of the product.
    * `Rate`: Numerical rating given by the customer (1-5 stars).
    * `Review`: The raw text of the customer review (primary focus for sentiment analysis).
    * `Summary`: A brief summary of the review.
    * `Sentiment`: A pre-labeled sentiment category (Positive, Negative, Neutral) for the review (not used for this app's direct sentiment analysis, but could be for model training).

    **Usage Tips:**
    * **Filters:** Use the sidebar filters to narrow down your analysis to specific products or sentiment types.
    * **Tabs:** Navigate through the tabs for different views of the data: overall metrics, sentiment distribution charts, product-specific insights, and custom text analysis.
    * **Raw Data:** You can view the raw filtered data at the bottom of the "Overview & Metrics" tab.
    """)

# Optional: Raw Data Table (can be placed inside a tab or expander as needed)
with st.expander("Show Raw Data Table (Filtered)"):
    st.subheader("Filtered Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)

st.info("💡 Pro Tip: Filter by product or sentiment type in the sidebar to get more specific insights!")