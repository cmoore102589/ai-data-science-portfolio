# E-Commerce Customer Review Classification Using RNN

## Project Overview
This project applies **Natural Language Processing (NLP)** and **deep learning** to classify customer reviews from an e-commerce dataset. The task is to predict whether a review recommends a product, based on customer-written feedback. This binary classification problem uses a Recurrent Neural Network (RNN) built with TensorFlow and trained on textual data extracted from product reviews.

## Dataset
- **Source**: Women’s Clothing E-Commerce Reviews
- **Target Variable**: `Recommended IND` (0 = Not Recommended, 1 = Recommended)
- **Textual Features Used**:
  - Title
  - Review Text
  - Division Name
  - Department Name
  - Class Name

## Tasks Completed
1. **Loaded** and preprocessed the dataset from CSV format.
2. **Concatenated** multiple text fields into a single `Reviews` feature.
3. **Cleaned** the combined reviews using regular expressions (lowercase, punctuation removal).
4. **Vectorized** the cleaned text using TensorFlow's `TextVectorization` layer.
5. **Built** a deep learning model using a **Bidirectional LSTM-based RNN**.
6. **Trained and validated** the model with a clear accuracy threshold.
7. **Evaluated model performance** and recommended deployment based on a predefined success criterion.

## Technologies Used
- Python (Pandas, NumPy)
- TensorFlow / Keras
- Regular Expressions (re)
- Matplotlib
- Scikit-learn (for data splitting)

## Model Architecture
- **Embedding Layer** with vocabulary size = 10,000 and embedding dimensions = 128
- **Bidirectional LSTM** layers:
  - First: 128 units with return sequences
  - Second: 64 units
- **Dropout**: 20% after each LSTM block
- **Dense Output**: 1 neuron with sigmoid activation (binary classification)

## Performance Summary

| Metric          | Value        |
|-----------------|--------------|
| Accuracy        | ~90%         |
| Threshold Met   | Yes (≥85%)   |
| Recommendation  | **Yes** — Model is recommended for deployment |

## Key Insight
Switching from a standard LSTM to a **Bidirectional LSTM** significantly improved model accuracy. This architecture captures contextual information from both directions, which is particularly beneficial for understanding customer reviews.

## Visualizations
- Training and validation accuracy plotted across epochs
- Test set evaluation output
