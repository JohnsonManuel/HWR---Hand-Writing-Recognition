# Handwriting Recognition (HWR) using Deep Learning

## Overview
This project implements a Handwriting Recognition (HWR) system using deep learning techniques, specifically Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), with the goal of recognizing handwritten names from images.

## Dataset
The dataset used for training and testing is sourced from [Kaggle Handwriting Recognition Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition), which includes handwritten names along with their corresponding images.

### Data Cleaning
The dataset underwent preprocessing steps, including handling unreadable labels and normalizing text.

## Implementation
### Technologies Used:
- TensorFlow and Keras for deep learning model development.
- OpenCV for image processing tasks.
- Matplotlib for visualization.

### Model Architecture
The neural network model architecture consists of:
- Convolutional layers for feature extraction.
- Bidirectional LSTM layers for sequence modeling.
- CTC (Connectionist Temporal Classification) loss function for sequence prediction.

### Preprocessing
Images were resized, normalized, and converted to grayscale before feeding into the model.

### Training
The model was trained on a dataset of 30,000 images with a validation set of 3,000 images for 60 epochs using the Adam optimizer with a learning rate of 0.0001.

## Files
- `HWR_model.keras`: Saved model file.
- `README.md`: Documentation for the project.

## Usage
To use the model:
1. Clone the repository.
2. Install the required dependencies (`tensorflow`, `opencv-python`, `matplotlib`).
3. Load the trained model (`HWR_model.keras`) for inference.

## Acknowledgments
- Kaggle Handwriting Recognition Dataset for providing the dataset.


