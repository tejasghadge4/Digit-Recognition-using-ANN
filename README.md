# Digit-Recognition-using-ANN

Overview
This project implements a digit recognition system using the MNIST dataset, a popular benchmark dataset for image processing and machine learning tasks. The goal of this project is to classify handwritten digits (0–9) using a Neural Network. The project demonstrates the application of deep learning for image recognition, preprocessing, training, evaluation, and visualization of predictions.

Features

1. Data Preprocessing:
-Normalized pixel values to the range [0, 1] for better model performance.
-Flattened images to a suitable format for neural network input.

2. Model Architecture:
-A simple Artificial Neural Network (ANN) with multiple dense layers.
-ReLU activation for hidden layers and softmax for output.

3. Training & Evaluation:
-Trained on the MNIST dataset with 60,000 images for training and 10,000 images for testing.
-Achieved high accuracy on test data.

4.Visualization:
-Displayed a subset of the dataset images with their corresponding labels.
-Visualized predicted results for individual test samples.

Dataset

The project uses the MNIST dataset, which contains:

-60,000 training samples: Images of handwritten digits (28x28 grayscale).
-10,000 testing samples: Separate set of images for evaluation.
-Each image is labeled with the corresponding digit (0–9).

Technologies Used:

1. Python: Programming language used to implement the project.
2. TensorFlow/Keras: Deep learning framework for building and training the neural network.
3. NumPy: For numerical computations and data manipulation.
4. Matplotlib: For visualizing images and predictions.

Model Architecture
The model consists of:

Input Layer:
Accepts 28x28 grayscale images (flattened to 784 input features).

Hidden Layers:
Layer 1: Dense layer with 128 neurons and ReLU activation.
Layer 2: Dense layer with 64 neurons and ReLU activation.

Output Layer:
Dense layer with 10 neurons (one for each digit) and softmax activation for probability distribution.
Results
Training Accuracy: ~98% (depending on epochs and parameters).
Test Accuracy: ~97% (on unseen data).
Example Prediction: The model can predict individual handwritten digits and displays both the image and its predicted label.
