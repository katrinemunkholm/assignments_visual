# Assignment 2 - Classification Benchmarks with Logistic Regression and Neural Networks


## Description

This repository contains scripts for classifying the CIFAR-10 dataset using two machine learning approaches: Logistic Regression and Neural Networks. The aim is to demonstrate how to implement these classifiers using scikit-learn and evaluate their performance through classification reports and loss curves.
For the CIFAR-10 dataset, these classifiers classify the images into 10 different categories, namely 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship' and 'truck'.


### Task Overview
For this assignment, the main objectives were to develop Python scripts capable of performing the following functions using the CIFAR-10 dataset:

1. Preprocess the data (CIFAR-10) by converting images to grayscale, normalizing the pixel values, and reshaping the data for model input requirements.
2. Implement and train a Logistic Regression classifier using scikit-learn, configuring it to handle multi-class classification.
3. Similarly, implement and train a Neural Network classifier, specifically a Multi-layer Perceptron (MLP), using scikit-learn. 
4. Generate a classification report for each model, detailing metrics such as accuracy, precision, recall, and F1-score, and save these reports.
5. For the Neural Network model, plot and save a curve of the loss values obtained during training to visualize model learning progress.


## Data Source

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, each with 6,000 images. The dataset is split into 50,000 training images and 10,000 testing images. The dataset can be downloaded from the CIFAR-10 [website](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
See file structure below. 

## Setup and Running the Analysis

### Requirements

- Python 3
- Libraries: 
`keras==3.3.3`
`matplotlib==3.8.4`
`numpy==1.26.4`
`pandas==2.2.2`
`scikit_learn==1.4.2`
`seaborn==0.13.2`
`tensorflow==2.16.1`


### Environment Setup and Execution

1. **Environment Setup**:
   To set up a virtual environment and install the required packages, run the following in the command line:
   ```bash
   bash setup.sh
   ```

2. **Running the Scripts**:
   - For Logistic regression classifier:
     ```bash
     bash runLogReg.sh
     ```
   - For Neural Network classifier:
     ```bash
     bash runMLP.sh
     ```

## Summary of Key Points from the Outputs

The classification reports and loss curve plots are saved in the `out` folder. The evaluation of the models highlights the following results:

1. **Logistic Regression Model**:
   - **Performance**: The Logistic Regression model exhibits relatively low performance with a weighted average F1-score of approximately 0.31.
   - **Analysis**: This indicates limited effectiveness in classifying the CIFAR-10 dataset using logistic regression.

2. **Multi-Layer Perceptron (MLP) Classifier**:
   - **Performance**: The MLP classifier demonstrates a slightly improved performance compared to Logistic Regression, with a weighted average F1-score of around 0.39.
   - **Loss Curve**: The loss curve shows a gradual decrease during training, indicating that the model is learning and improving over the iterations.

Overall, the MLP classifier outperforms the Logistic Regression model, but both models indicate room for improvement in classification performance on the CIFAR-10 dataset.


## Discussion of Limitations and Possible Steps to Improvement

The models used in this project provide baseline performances. For enhanced results, employing deeper architectures like Convolutional Neural Networks (CNNs) could be explored. Additionally, experimenting with different hyperparameters and preprocessing techniques might yield better classification accuracy.

### Future Steps

- Implement CNNs to potentially improve performance.
- Adjust hyperparameters such as learning rate and batch size.
- Explore advanced data augmentation techniques to improve model robustness.

## File Structure

The downloaded folder should be placed in the "in" folder and named "cifar10". 
The scripts assume for following structure of the repository:
```
assignment2/
│
├── in/
│   └── cifar10/
│       └── ...
└── out/
    └── Logistic_regression_classification_report.csv 
    └── MLP_classification_report.csv 
    └── loss_curve_MLP.png
│
├── src/
│   └── Logistic_reg_classifier.py
│   └── Neural_network_classifier.py
│   └──utils
│
├── README.md
├── requirements.txt
├── setup.sh
├── runLogReg.sh
└── runMLP.sh

```