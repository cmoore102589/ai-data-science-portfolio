# Project 7 – Final COVID-19 Sentiment and Topic Modeling Analysis

## Overview
This project involves advanced natural language processing (NLP) and machine learning techniques to analyze and predict sentiments in tweets related to COVID-19. It uses the `Corona_NLP_train.csv` and `Corona_NLP_test.csv` datasets, which include tweet content along with associated metadata and labeled sentiments.

## Objectives
- Preprocess and clean tweets from multiple metadata sources
- Perform POS tagging and named entity recognition
- Analyze text length and word counts using various visualizations
- Conduct TF-IDF and frequency-based analysis
- Build and evaluate classification models for sentiment analysis
- Apply topic modeling (LDA) to discover latent themes in tweets
- Visualize semantic distributions using PCA

## Key Steps
1. **Data Cleaning**:
   - Removed dates, URLs, hashtags, usernames, and special characters.
   - Filtered short words and empty rows.
2. **NLP Processing**:
   - Part-of-speech tagging and Named Entity Recognition (NER)
   - Visualization of dependency trees and specific entities like GEOLOCATION and MONEY
3. **Tokenization & Lemmatization**:
   - Converted raw text to tokens and extracted lemmatized versions
4. **Text Analysis & Visualizations**:
   - Visualized sentiment distributions, text length, and word counts
   - Identified top unigrams and bigrams using TF-IDF
   - Used Scattertext for sentiment token analysis
5. **Feature Engineering**:
   - Applied CountVectorizer and TfidfVectorizer with n-grams
   - Calculated cosine similarity between tweets
   - Computed average corpus vector
6. **Modeling**:
   - Built and evaluated three sentiment classification models using Random Forest, Logistic Regression, and SVM
   - Tuned hyperparameters with GridSearchCV
7. **Topic Modeling**:
   - Constructed LDA models using both CountVectorizer and TfidfVectorizer
   - Visualized topic distributions and dimensionality with PCA

## Results
- Achieved solid model performance across classifiers, with Random Forest showing high interpretability.
- Discovered clear topic separations in COVID-19 discourse.
- Gained meaningful insights from NER and token frequency analysis.

## Practical Applications
This project has valuable applications in the **healthcare, government, and social media sectors**:
- **Public Health Monitoring**: Real-time detection of sentiment trends around policies, vaccines, and virus-related events.
- **Crisis Communication**: Organizations can tailor messaging based on prevailing public sentiment.
- **Information Surveillance**: Detection of misinformation or public anxiety spikes through topic modeling and sentiment tracking.
- **Market Research**: Healthcare providers and pharmaceutical companies can use this model to understand public reactions to products or campaigns during pandemics.

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn, SpaCy, Matplotlib, Seaborn, Scattertext, PyLDAvis)
- Jupyter Notebook
- NLP: TfidfVectorizer, CountVectorizer, LDA, SpaCy NER
- ML Models: Logistic Regression, Random Forest, Support Vector Machine

## Files
- `Corona_NLP_train.csv`
- `Corona_NLP_test.csv`
- `Project7_Moore_Matthew - Txt Mining.ipynb`
