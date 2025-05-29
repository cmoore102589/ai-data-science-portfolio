# Project 7: Sentiment Analysis with Sentiment140

## Overview
This project implements a sentiment classification system using the Sentiment140 dataset, which contains tweets labeled with sentiment polarity: positive, neutral, or negative. The primary goal is to predict the sentiment of a tweet based on its text content using machine learning models.

## Dataset
- **Source**: Sentiment140 (Stanford University)
- **Format**: CSV file with six columns (polarity, id, date, query, user, tweet text)
- **Polarity Labels**: 
  - `0` = Negative
  - `2` = Neutral
  - `4` = Positive

## Objectives
1. Preprocess and clean tweet text.
2. Train a baseline sentiment model using Support Vector Machines (SVM).
3. Tune hyperparameters using GridSearchCV.
4. Compare alternative classifiers including Logistic Regression and Random Forest.
5. Evaluate models using accuracy, precision, recall, and F1-score.

## Technologies Used
- Python
- Scikit-learn
- NLTK
- SpaCy
- Pandas
- Matplotlib

## Practical Application
Sentiment analysis using general tweets can significantly enhance understanding of public perception across a wide range of industries. Potential applications include:

- **Customer Service**: Automatically identify dissatisfied customers through negative tweets to trigger customer support interventions.
- **Brand Monitoring**: Companies can monitor public sentiment in real time to gauge reactions to marketing campaigns or PR events.
- **Financial Services**: Analysts can use sentiment trends on social media to predict consumer confidence and market sentiment.
- **Product Feedback**: Real-time analysis of public feedback on product launches or updates via social media platforms.

The machine learning pipeline built here can be integrated into real-world systems for real-time sentiment classification and analytics dashboards.

## Status
Complete. Ready for deployment or extension into a real-time classification system.
