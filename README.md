# Document Type Classifier

## Overview

This project implements a Document Type Classification system using a pre-trained EfficientNet-B0 model. The objective is to classify document images into one of the 16 document categories from the RVL-CDIP dataset.

## Dataset

* Dataset: RVL-CDIP
* Total Classes: 16
* Image Type: Grayscale Document Images

## Model

* Architecture: EfficientNet-B0 (Transfer Learning)
* Framework: PyTorch
* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Learning Rate: 0.0001
* Batch Size: 45
* Epochs: 5

## Preprocessing

* Grayscale to RGB conversion
* Resize to 224×224
* Random Rotation
* Normalization using ImageNet statistics

## Results

| Metric            | Value |
| ----------------- | ----- |
| Test Accuracy     | 5.68% |
| Macro F1 Score    | 0.01  |
| Weighted F1 Score | 0.01  |

### Observation

The model successfully completed training and evaluation. However, the current configuration achieved limited classification performance and showed a tendency to predict a dominant class for most samples. Further improvements in data preparation, training strategy, and hyperparameter tuning are required to improve accuracy.

## Technologies Used

* Python
* PyTorch
* TorchVision
* Scikit-Learn
* CUDA

## Future Improvements

* Train for more epochs
* Hyperparameter tuning

![image alt](https://github.com/starAditya/ML/blob/09746cd9255cf81e9f882436e278a8569273a88d/final%20test%201.jpeg)
![image alt](https://github.com/starAditya/ML/blob/71886084a59712c476cae0ce077df3e61113ea77/Final%20test%20Accuracy%202.jpeg)
![image alt](https://github.com/starAditya/ML/blob/bd229291d1112af1fabf7b56fa5c34b9607a010f/final%20test%20Accuracy%203.jpeg)

