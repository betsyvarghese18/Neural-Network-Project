# Neural-Network-Project

Medical Image Diagnosis with CNN
Project Overview
This project focuses on automated medical image diagnosis using Convolutional Neural Networks (CNNs) and traditional machine learning models with HOG (Histogram of Oriented Gradients) features. The goal is to accurately classify medical images into their respective disease categories, improving diagnosis speed and reliability.

We compare deep learning (EfficientNetV2B0) against classical ML models (SVM, Random Forest, KNN) to evaluate performance on our dataset.

Objectives
Develop a robust image classification system for medical diagnosis.

Compare CNN-based deep learning performance with classical ML models using HOG features.

Evaluate models on accuracy, F1-score, and confusion matrices.

Technologies Used
Language: Python

Libraries:

Deep Learning: TensorFlow, Keras

Image Processing: OpenCV, Pillow, scikit-image

ML Models: scikit-learn

Visualization: Matplotlib, Seaborn

Methodology
1️. Data Preprocessing
Image resizing (CNN: 224x224, HOG: 128x128)

Normalization & augmentation (flip, rotation, zoom, contrast adjustments)

One-hot encoding for CNN labels

2️. Feature Extraction
CNN Pipeline: Direct learning from pixel data via EfficientNetV2B0

HOG Features: Edge-based feature extraction for SVM, RF, and KNN

3️. Model Training
Deep Learning (CNN)
Base Model: EfficientNetV2B0 (pre-trained on ImageNet)

Global Average Pooling, Dropout, Dense (Softmax) layers

Optimizer: Adam

Loss: Categorical Crossentropy

EarlyStopping, ReduceLROnPlateau, ModelCheckpoint for training optimization

Fine-tuning last 25 layers for improved performance

Classical ML Models (HOG Features)
SVM: RBF/Linear kernels, tuned C and gamma

Random Forest: Tuned n_estimators and max_depth

KNN: Tuned n_neighbors and weights

Hyperparameter tuning with GridSearchCV

 Model Evaluation Metrics
Accuracy

Macro & Weighted F1-score

Confusion Matrix (normalized)
