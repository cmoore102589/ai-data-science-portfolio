# Project 8: COVID-19 Tweet Sentiment Analysis and Topic Modeling

## Overview
This project analyzes public sentiment and thematic trends in tweets related to the COVID-19 pandemic. Using natural language processing (NLP) and machine learning, we clean, preprocess, and classify tweets while also uncovering underlying topics through topic modeling.

## Dataset
- **Source**: [Kaggle: Corona_NLP_train.csv & Corona_NLP_test.csv](https://www.kaggle.com/datatasks/covid-19-nlp-text-classification)
- **Fields**: `UserName`, `ScreenName`, `Location`, `TweetAt`, `OriginalTweet`, `Sentiment`
- **Sentiment Labels**:
  - `Positive`
  - `Negative`
  - `Neutral`
  - `Extremely Positive`
  - `Extremely Negative`

## Objectives
1. Merge relevant fields to form tweet context (`Location`, `TweetAt`, `OriginalTweet`)
2. Clean and preprocess tweets (remove links, hashtags, special characters, etc.)
3. Extract and lemmatize tokens
4. Perform exploratory data analysis:
   - Sentiment distribution
   - Tweet length and word count visualization
   - Top unigrams and bigrams by TF-IDF
   - Token frequency and scores using `scattertext`
5. Feature extraction using:
   - CountVectorizer
   - TfidfVectorizer
   - Cosine similarity
6. Classification Models:
   - Random Forest (baseline)
   - Random Forest with hyperparameter tuning (GridSearchCV)
   - Logistic Regression, SVM, and Random Forest comparison
7. Topic Modeling:
   - LDA with CountVectorizer and TfidfVectorizer
   - Visualization of top words per topic
   - PCA-based topic space visualization

## Technologies Used
- Python
- Scikit-learn
- NLTK & SpaCy
- Seaborn & Matplotlib
- Scattertext
- pyLDAvis
- Pandas & NumPy

## Real-World Applications
- **Health Policy Insights**: Track changes in public sentiment regarding vaccination, lockdowns, and public health messaging.
- **Social Media Monitoring**: Identify misinformation trends or geographic hot spots based on tweet content.
- **Crisis Communication**: Governments and NGOs can tailor communication strategies based on sentiment shifts and topic focus.
- **Academic Research**: Provides a case study in real-time natural language understanding during a global health crisis.

## Results Summary
- Preprocessing led to improved text clarity and reduced noise.
- Topic models revealed clear clusters around health concerns, government policy, and pandemic coping.

## Status
Complete. Ready for deployment, extension into time-series tweet tracking, or integration into a COVID-19 monitoring dashboard.
