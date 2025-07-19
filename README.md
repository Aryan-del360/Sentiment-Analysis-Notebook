<<<<<<< HEAD
# 🛍️ E-commerce Product Review Sentiment Analysis

## Project Overview

This project focuses on performing sentiment analysis on a dataset of e-commerce product reviews. The goal is to classify customer reviews into positive, negative, or neutral sentiments using natural language processing (NLP) techniques and pre-trained lexicon-based models (VADER and TextBlob). This analysis provides valuable insights into customer satisfaction and product perception, which can inform business decisions.

## Problem Statement

In the vast landscape of e-commerce, understanding customer feedback is paramount for business growth and product improvement. Manually sifting through thousands of reviews is impractical. This project addresses the challenge of automatically identifying the sentiment expressed in product reviews, allowing businesses to quickly gauge public opinion, pinpoint popular features, and identify areas requiring attention.

## Dataset

The dataset used in this project is `Dataset-SA.csv`. It contains comprehensive information about e-commerce product reviews, including:

* `product_name`: Name of the reviewed product.
* `product_price`: Price of the product.
* `Rate`: Numerical rating given by the customer (1-5 stars).
* `Review`: The raw text of the customer review (primary focus for sentiment analysis).
* `Summary`: A brief summary of the review.
* `Sentiment`: A pre-labeled sentiment category (Positive, Negative, Neutral) for the review, serving as our ground truth.

**Key Dataset Characteristics:**
* **Size:** Over 200,000 product reviews.
* **Imbalance:** The dataset exhibits a significant positive sentiment bias, with a majority of reviews classified as 'Positive' and high star ratings (4 and 5 stars).
* **Review Length:** Reviews are generally concise, with an average length of around 11 characters.

## Methodology & Project Pipeline

The project follows a structured data science pipeline:

1.  **Project Setup & Data Acquisition:**
    * Imported essential libraries (Pandas, NumPy, Matplotlib, Seaborn, NLTK, TextBlob, Scikit-learn).
    * Downloaded necessary NLTK data resources (`vader_lexicon`, `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`).
    * Loaded the `Dataset-SA.csv` into a Pandas DataFrame.

2.  **Exploratory Data Analysis (EDA):**
    * Performed data integrity checks, identifying and handling missing values in the `Review` column.
    * Converted the `Rate` column to a numeric data type, dropping invalid entries.
    * Analyzed the distribution of review lengths.
    * Visualized the distribution of product ratings to understand overall customer satisfaction.

3.  **Text Preprocessing (NLP):**
    * Cleaned the raw `Review` text by converting to lowercase, removing URLs, HTML tags, punctuation, numbers, and extra whitespaces.
    * Applied **tokenization** to break reviews into individual words.
    * Performed **stopword removal** to eliminate common, less informative words.
    * Applied **lemmatization** to reduce words to their base forms (e.g., "running" to "run").
    * Created a new `cleaned_review` column in the DataFrame.

4.  **Sentiment Analysis with VADER and TextBlob:**
    * Used **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to generate compound sentiment scores and classify reviews as 'Positive', 'Negative', or 'Neutral'.
    * Employed **TextBlob** to calculate polarity (sentiment strength) and subjectivity scores, and classified sentiments similarly.

5.  **Model Evaluation and Comparison:**
    * Compared the sentiments predicted by VADER and TextBlob against the `Sentiment` (true) labels provided in the dataset.
    * Calculated standard classification metrics: **Accuracy, Precision, Recall, and F1-score**.
    * Generated **confusion matrices** to visualize true vs. predicted classifications for both models.

6.  **Visualization of Results and Insights:**
    * Generated various plots to visualize:
        * Distribution of true sentiments.
        * Distributions of VADER and TextBlob predicted sentiments.
        * Comparative bar plots of true vs. predicted sentiment proportions.
        * Sentiment distribution by product rating (for true, VADER, and TextBlob sentiments).

## Key Findings & Insights

* **Positive Dominance:** The dataset is heavily skewed towards positive reviews (both in true sentiment and star ratings), making it easier for models to achieve higher accuracy on the 'Positive' class.
* **VADER's Performance:** VADER achieved an overall accuracy of approximately **67.18%**. It showed strong performance in identifying positive reviews (F1-score ~0.81).
* **TextBlob's Performance:** TextBlob performed slightly lower with an overall accuracy of approximately **62.90%**, exhibiting similar strengths and weaknesses to VADER.
* **Challenges with Neutral & Negative:** Both lexicon-based models struggled significantly with accurately classifying **Neutral** reviews (very low F1-scores) and showed lower recall for **Negative** reviews. This is a common limitation for these types of models, especially with an imbalanced dataset where 'neutral' might be ambiguous.
* **Rating-Sentiment Alignment:** Visualizations confirmed a strong correlation between higher star ratings and positive sentiment, and lower ratings with negative/neutral sentiments.

## Technologies Used

* **Python**
* **Pandas** (for data manipulation and analysis)
* **NumPy** (for numerical operations)
* **NLTK (Natural Language Toolkit)** (for preprocessing and VADER)
* **TextBlob** (for sentiment analysis)
* **Scikit-learn** (for evaluation metrics like `accuracy_score`, `classification_report`, `confusion_matrix`)
* **Matplotlib** (for plotting)
* **Seaborn** (for enhanced visualizations)

## Future Enhancements

This project serves as a strong foundation. Potential improvements include:

* **Advanced Supervised Models:** Training machine learning models (e.g., Logistic Regression, SVM, XGBoost) or deep learning models (e.g., LSTMs, Transformers like BERT) on the labeled data after feature engineering (e.g., TF-IDF, Word Embeddings) for potentially higher accuracy, especially for minority classes.
* **Addressing Class Imbalance:** Implementing techniques like oversampling (SMOTE) or undersampling to improve the models' ability to classify negative and neutral reviews.
* **Hyperparameter Tuning:** Optimizing thresholds for VADER/TextBlob or parameters for advanced ML models.
* **Error Analysis:** Conducting a detailed qualitative analysis of misclassified reviews to understand model limitations and guide further improvements.
* **Aspect-Based Sentiment Analysis:** Extending the analysis to identify sentiment towards specific product features (e.g., "camera," "battery life") within reviews.
* **Deployment:** Developing a simple web application (using Streamlit/Flask) to allow users to input text and get real-time sentiment predictions.

models (e.g., Logistic Regression, SVM, XGBoost) or deep learning models (e.g., LSTMs, Transformers like BERT) on the labeled data after feature engineering (e.g., TF-IDF, Word Embeddings) for potentially higher accuracy, especially for minority classes.
* **Addressing Class Imbalance:** Implementing techniques like oversampling (SMOTE) or undersampling to improve the models' ability to classify negative and neutral reviews.
* **Hyperparameter Tuning:** Optimizing thresholds for VADER/TextBlob or parameters for advanced ML models.
* **Error Analysis:** Conducting a detailed qualitative analysis of misclassified reviews to understand model limitations and guide further improvements.
* **Aspect-Based Sentiment Analysis:** Extending the analysis to identify sentiment towards specific product features (e.g., "camera," "battery life") within reviews.
* **Deployment:** Developing a simple web application (using Streamlit/Flask) to allow users to input text and get real-time sentiment predictions.
---
