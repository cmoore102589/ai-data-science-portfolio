# Handwritten Digit Generation with DCGAN

## Project Overview
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic images of handwritten digits. Using the MNIST dataset, a generator and discriminator network are trained adversarially. The generator learns to produce fake digits while the discriminator learns to distinguish between real and fake images.

The project culminates in the creation of a **GIF animation** that visualizes the generator's progress over epochs.

## Dataset
- **Source**: [MNIST Dataset](https://keras.io/api/datasets/mnist/)
- **Shape**: 28x28 grayscale images
- **Classes**: Digits 0–9
- **Total Images**: 60,000 (training set)

## Project Goals
- Load and preprocess the MNIST dataset
- Construct a **Generator** and **Discriminator**
- Train a DCGAN model using adversarial training
- Monitor image outputs and generate a visual GIF timeline

## Technologies Used
- Python
- TensorFlow & Keras
- NumPy
- Matplotlib & ImageIO
- PIL (Python Imaging Library)

## DCGAN Architecture

### Generator
- Dense layer + Reshape
- Conv2DTranspose layers (upsampling)
- Final layer with `sigmoid` activation for grayscale output

### Discriminator
- Conv2D layers (downsampling)
- LeakyReLU activations
- Dropout for regularization
- Final Dense layer with `sigmoid` for binary classification

## Training Summary

| Component        | Details                     |
|------------------|-----------------------------|
| Latent Dimension | 128                         |
| Batch Size       | 32                          |
| Optimizer        | Adam                        |
| Loss Function    | Binary Crossentropy         |
| Epochs           | 50                          |
| Callback         | Saves sample images per epoch |

## GIF Output
- A custom Keras callback saves generated images during training.
- After training, all saved images are compiled into a `generated_images.gif`.
- The GIF visually illustrates the improvement of the generated digits over time.
