import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Configuration & Setup ---
# 1. Page Config: Modern, Wide Layout, Initial Sidebar State
st.set_page_config(
    page_title="Product Trend Analyzer 🚀",
    page_icon="📈",
    layout="wide", # Use 'wide' layout for more space
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

# Set up matplotlib style for a cleaner look
plt.style.use('seaborn-v0_8-darkgrid') # Or 'ggplot', 'fivethirtyeight'
# Adjust plot defaults for better aesthetics
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# --- Helper Functions ---
@st.cache_data # Cache data loading to improve performance
def load_and_process_data():
    """Generates and processes synthetic product sales data."""
    np.random.seed(42) # for reproducibility

    # Generate synthetic data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    for date in dates:
        for product in products:
            for region in regions:
                sales = np.random.randint(50, 500) + np.random.rand() * 50 # Base sales
                price = np.random.uniform(10, 100)
                if 'A' in product: sales += np.random.randint(0, 100) # Product A bonus
                if date.month in [11, 12]: sales += np.random.randint(0, 150) # Year-end boost
                
                data.append([date, product, region, sales, price, sales * price]) # Add Revenue

    df = pd.DataFrame(data, columns=['Date', 'Product', 'Region', 'Sales_Units', 'Price_Per_Unit', 'Revenue'])
    
    df['Month'] = df['Date'].dt.to_period('M')
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.to_period('Q')
    
    return df

# Load the data
df = load_and_process_data()

# --- Title & Introduction (Gen Z friendly, concise) ---
st.title("🛍️ Product Performance Dashboard")
st.markdown("""
Welcome to your go-to hub for understanding product trends! 🚀
Dive into sales, revenue, and regional performance with sleek visualizations and quick insights.
Let's get those data-driven vibes! ✨
""")

# --- Sidebar Filters (Intuitive, clean) ---
st.sidebar.header("🎯 Filter Your View")

all_products = ['All Products'] + list(df['Product'].unique())
selected_product = st.sidebar.selectbox(
    "Choose a Product:",
    options=all_products,
    index=0 # 'All Products' selected by default
)

all_regions = ['All Regions'] + list(df['Region'].unique())
selected_region = st.sidebar.selectbox(
    "Select a Region:",
    options=all_regions,
    index=0
)

# Date Range Slider
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Pick a Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Ensure date_range always has two elements
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
elif len(date_range) == 1:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[0]) # If only one date selected, make it a single day range
else: # Default to full range if nothing selected or invalid
    start_date = pd.to_datetime(min_date)
    end_date = pd.to_datetime(max_date)

# Apply filters
filtered_df = df[
    (df['Date'] >= start_date) & 
    (df['Date'] <= end_date)
]

if selected_product != 'All Products':
    filtered_df = filtered_df[filtered_df['Product'] == selected_product]

if selected_region != 'All Regions':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("😬 No data available for the selected filters. Try adjusting your selections!")
else:
    # --- Key Performance Indicators (KPIs) ---
    st.markdown("---") # Visual separator
    st.header("📊 Quick Stats")

    total_sales = filtered_df['Sales_Units'].sum()
    total_revenue = filtered_df['Revenue'].sum()
    avg_price = filtered_df['Price_Per_Unit'].mean()
    num_products = filtered_df['Product'].nunique()
    num_regions = filtered_df['Region'].nunique()

    col1, col2, col3, col4, col5 = st.columns(5) # Use columns for a neat layout
    with col1:
        st.metric(label="Total Sales Units", value=f"{total_sales:,.0f} 📈")
    with col2:
        st.metric(label="Total Revenue", value=f"${total_revenue:,.2f} 💰")
    with col3:
        st.metric(label="Avg. Price/Unit", value=f"${avg_price:,.2f} 🏷️")
    with col4:
        st.metric(label="Unique Products", value=f"{num_products} 📦")
    with col5:
        st.metric(label="Unique Regions", value=f"{num_regions} 🌍")

    # --- Visualizations (Professional, insight-driven) ---
    st.markdown("---")
    st.header("✨ Visual Insights")

    # Sales Trend Over Time
    st.subheader("1. Sales Performance Over Time")
    sales_over_time = filtered_df.groupby('Date')['Sales_Units'].sum().reset_index()
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=sales_over_time, x='Date', y='Sales_Units', ax=ax1, marker='o')
    ax1.set_title('Total Sales Units Over Time 📊')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales Units')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    st.markdown("""
    **Insight Summary:**
    * **Trend Spotting:** Observe the overall trajectory of sales. Is it growing, declining, or stable?
    * **Seasonal Patterns:** Look for recurring peaks (e.g., year-end holidays) or dips (e.g., off-season).
    * **Anomalies:** Identify any sudden, unusual spikes or drops that might indicate specific events (promotions, outages).
    """)

    # Sales by Product
    st.subheader("2. Top Product Sales (Units)")
    sales_by_product = filtered_df.groupby('Product')['Sales_Units'].sum().sort_values(ascending=False).reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(data=sales_by_product, x='Sales_Units', y='Product', ax=ax2, palette='viridis')
    ax2.set_title('Sales Units by Product Category 📦')
    ax2.set_xlabel('Total Sales Units')
    ax2.set_ylabel('Product')
    st.pyplot(fig2)
    st.markdown("""
    **Insight Summary:**
    * **Top Performers:** Quickly identify which products are driving the most sales.
    * **Underperformers:** See if any products are lagging behind and might need attention.
    * **Product Mix:** Understand the relative contribution of each product to overall unit sales.
    """)

    # Revenue by Region
    st.subheader("3. Revenue Distribution by Region")
    revenue_by_region = filtered_df.groupby('Region')['Revenue'].sum().sort_values(ascending=False).reset_index()
    fig3, ax3 = plt.subplots()
    sns.pie(data=revenue_by_region, x='Revenue', labels=revenue_by_region['Region'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax3.set_title('Revenue Share by Region 🌍')
    ax3.set_ylabel('') # Hide default y-label for pie chart
    st.pyplot(fig3)
    st.markdown("""
    **Insight Summary:**
    * **Regional Dominance:** See which regions are generating the most revenue.
    * **Market Share:** Understand the proportional revenue contribution of each geographical area.
    * **Targeted Strategies:** Helps in allocating resources or tailoring marketing efforts based on regional performance.
    """)
    
    # --- Detailed Data View (Optional, for deeper dive) ---
    st.markdown("---")
    st.header("🔍 Dive Deeper: Raw Data View")
    if st.checkbox("Show Detailed Data Table"):
        st.dataframe(filtered_df)

    st.markdown("""
    ---
    Got questions? Feel free to reach out or explore the data further! 🧑‍💻
    """)