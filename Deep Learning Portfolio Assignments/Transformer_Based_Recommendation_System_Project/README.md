# Transformer-Based Movie Recommendation System (BST Model)

## Project Overview
This project implements a **Transformer-based recommendation system** using the MovieLens 1M dataset. The goal is to predict user ratings of movies based on historical behavior sequences. The solution adapts and extends the **Behavior Sequence Transformer (BST)** model to incorporate both sequential movie rating data and metadata like user demographics and movie genres.

## Dataset
- **Source**: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
- **Data Used**:
  - User demographics: gender, age group, occupation
  - Movie metadata: title, genres
  - Ratings: 1–5 stars with timestamps
- **Size**:
  - 1 million ratings
  - 6,000 users
  - 4,000 movies

## Key Features
- Sequence modeling using Transformer attention mechanisms.
- User-aware and context-aware modeling through additional metadata.
- Train/test datasets built using sliding windows over timestamped user behavior.

## Tasks Completed
1. Downloaded and preprocessed the MovieLens dataset.
2. Created movie rating sequences per user (e.g., 6 previous movies → 7th prediction).
3. Engineered metadata including:
   - One-hot encoded genres
   - User categorical variables (sex, age group, occupation)
4. Created `tf.data.Dataset` objects for efficient training.
5. Built a Behavior Sequence Transformer (BST) model using Keras:
   - Multi-head attention over sequences
   - Dense layers for regression
6. Trained and evaluated the model with Mean Absolute Error (MAE) as the key metric.

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (for preprocessing)
- MultiHeadAttention from TensorFlow layers

## Model Architecture

| Component                | Details                                 |
|--------------------------|-----------------------------------------|
| Input Sequence           | Last 5 movies watched by a user         |
| Embeddings               | Movie IDs, ratings, user metadata       |
| Transformer Block        | Multi-head self-attention + normalization |
| Dense Layers             | Fully connected (32 → 16 units)         |
| Output                   | Regression score for next rating        |

## Evaluation

| Metric      | Value     |
|-------------|-----------|
| Loss        | MSE       |
| Evaluation  | MAE       |
| Optimizer   | Adagrad   |

Final model performance is reported via MAE on test set.
