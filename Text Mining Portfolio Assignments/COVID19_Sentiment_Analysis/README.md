# Project 8 – COVID-19 Tweet Analysis and Classification

## Overview

This project performs end-to-end text analysis and sentiment classification on COVID-19 tweets. It includes:
- Text cleaning and tokenization
- Part-of-speech tagging and named entity recognition (NER)
- TF-IDF vectorization and topic modeling (LDA)
- Sentiment classification using Logistic Regression, Random Forest, and SVM with GridSearchCV tuning
- Data visualization using PCA and scattertext

## Objectives

- Preprocess and analyze COVID-19 tweet content
- Build classification models to predict sentiment
- Explore topics using LDA on both CountVectorizer and TF-IDF representations
- Visualize latent topics and feature importance

## Practical Applications

This project reflects how NLP can assist **public health, policy, and communication teams** by:

- **Tracking public sentiment during health crises**: Enables faster understanding of public reactions to announcements, restrictions, or events.
- **Identifying misinformation**: Topic modeling can surface emerging concerns or conspiracy trends.
- **Resource planning**: Analyzing geographic or temporal trends in sentiment can help allocate medical resources more effectively.
- **Enhancing digital health communication**: Insights can guide more empathetic and effective messaging strategies.

This analysis framework is adaptable to other **public sector or social media analytics** challenges, especially in domains requiring sentiment tracking over time.

## Technologies Used

- Python
- Pandas, Scikit-learn, spaCy, Scattertext, Matplotlib, Seaborn
- Latent Dirichlet Allocation (LDA)
- Jupyter Notebook
