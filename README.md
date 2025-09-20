#  Osteoarthritis Severity Classification Using CNN (AlexNet-Based)

This repository contains the implementation of a deep learning pipeline to classify medical images of osteoarthritis severity into five categories: Doubtful, Mild, Moderate, Normal, and Severe. The project demonstrates baseline CNN modeling, model enhancements with fine-tuning, and automated hyperparameter optimization using Keras Tuner.

## Project Overview

The main objective is to develop a Convolutional Neural Network (CNN) model from scratch following AlexNet architecture principles (no pretrained weights) to perform multi-class classification of osteoarthritis severity using image data. The project covers data preprocessing, exploratory data analysis (EDA), baseline model creation, model refinement, hyperparameter tuning, and evaluation using robust classification metrics.

---

## Dataset

- Source: Osteoarthritis image dataset with train, validation (15% split), and test sets.
- Image size: 224x224 pixels RGB.
- Distribution: Five classes - Doubtful, Mild, Moderate, Normal, Severe.
- Exploratory Data Analysis (EDA) included visualization of sample images, histogram of pixel intensity distributions, brightness variability, and label distribution across datasets.

---

## Methodology

### Data Preprocessing & Augmentation
- Image resizing, random horizontal flipping, contrast modification, and per-image standardization to enhance model robustness and convergence speed.

### Model Architectures

#### Baseline Model
- Manual implementation of AlexNet-inspired CNN.
- Architecture: Multiple Conv2D layers, MaxPooling, Flatten, Dense layers with ReLU activations and Dropout.
- Compiled with Adam optimizer, categorical cross-entropy loss, evaluated with accuracy.

#### Modified Model
- Enhanced CNN with Batch Normalization to stabilize input distribution for each layer.
- Added Dropout to reduce overfitting.
- Used learning rate scheduler ReduceLROnPlateau to adjust learning rate dynamically during training.

#### Hyperparameter Tuning for Baseline Model
- Used Keras Tuner with Random Search to optimize convolutional filters, dense layer units, dropout rates, and learning rates.

---

## Results

### Baseline Model Performance
- Accuracy: ~20%
- Model struggled with detecting patterns effectively, showing low precision, recall, and F1-scores overall.

### Modified Model Performance
- Accuracy improved to ~64%
- Significant improvement in precision, recall, and F1-scores on most classes, except some like Normal class which still showed weaknesses.

### Hyperparameter Tuned (Base/Unmodified) Model Performance
- Accuracy: ~59%
- Balanced precision and recall in multiple classes, with especially strong performance on Severe class.
