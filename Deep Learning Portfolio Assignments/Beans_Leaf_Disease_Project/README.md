# Beans Leaf Disease Classification with CNN and Transfer Learning

## Project Overview
This project involves classifying **bean leaf diseases** using computer vision and deep learning techniques. The goal is to distinguish between three classes of bean leaves (healthy, angular leaf spot, and bean rust) using convolutional neural networks (CNN), data augmentation, and transfer learning.

The project evaluates three approaches:
1. A custom-built CNN.
2. CNN with data augmentation.
3. A transfer learning model using MobileNetV2.

## Dataset
- **Source**: TensorFlow Datasets (TFDS) – `beans` dataset
- **Classes**: 3
  - Angular Leaf Spot
  - Bean Rust
  - Healthy
- **Image Format**: RGB
- **Image Shape**: Resized to (128, 128)

## Tasks Completed
1. Loaded and normalized the dataset using TensorFlow Datasets.
2. Built a baseline CNN model with three convolutional layers.
3. Evaluated performance using accuracy and loss metrics.
4. Applied data augmentation techniques to reduce overfitting and improve model generalization.
5. Implemented a transfer learning pipeline using the pre-trained MobileNetV2.
6. Compared model performance across all three methods.

## Technologies Used
- Python
- TensorFlow / Keras
- TensorFlow Datasets (TFDS)
- Matplotlib for visualization

## Model Performance

| Model Type            | Test Accuracy | Notes                             |
|-----------------------|---------------|-----------------------------------|
| Baseline CNN          | ~0.82         | Some overfitting observed         |
| CNN with Augmentation | ~0.86         | Generalization improved           |
| Transfer Learning     | ~0.91         | Best performance, stable training |

## Key Findings
- The baseline CNN model trained well but showed signs of overfitting.
- Data augmentation improved generalization by increasing test accuracy.
- Transfer learning with MobileNetV2 outperformed both models, improving accuracy by over 9%.

## Folder Structure
