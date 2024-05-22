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

## Analysis

The evaluation of the models highlights the following results:

### Logistic Regression Model

- **Performance**: The Logistic Regression model exhibits relatively low performance with a weighted average F1-score of approximately 0.31.
- **Analysis**: This indicates limited effectiveness in classifying the CIFAR-10 dataset using logistic regression.

### Multi-Layer Perceptron (MLP) Classifier

- **Performance**: The MLP classifier demonstrates a slightly improved performance compared to Logistic Regression, with a weighted average F1-score of around 0.39.
- **Loss Curve**: The loss curve shows a gradual decrease during training, indicating that the model is learning and improving over the iterations.

Overall, the MLP classifier outperforms the Logistic Regression model, but both models indicate room for improvement in classification performance on the CIFAR-10 dataset.

### Classification Report: Logistic Regression

|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| airplane    | 0.3206    | 0.395  | 0.3539   | 1000    |
| automobile  | 0.3710    | 0.394  | 0.3822   | 1000    |
| bird        | 0.2570    | 0.210  | 0.2312   | 1000    |
| cat         | 0.2140    | 0.144  | 0.1721   | 1000    |
| deer        | 0.2450    | 0.195  | 0.2171   | 1000    |
| dog         | 0.3045    | 0.331  | 0.3172   | 1000    |
| frog        | 0.2947    | 0.285  | 0.2898   | 1000    |
| horse       | 0.3034    | 0.284  | 0.2934   | 1000    |
| ship        | 0.3379    | 0.395  | 0.3642   | 1000    |
| truck       | 0.3751    | 0.473  | 0.4184   | 1000    |
| **accuracy**| 0.3106    | 0.3106 | 0.3106   | 10000   |
| macro avg   | 0.3023    | 0.3106 | 0.3040   | 10000   |
| weighted avg| 0.3023    | 0.3106 | 0.3040   | 10000   |

### Classification Report: MLP

|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| airplane    | 0.4173    | 0.474  | 0.4438   | 1000    |
| automobile  | 0.4906    | 0.470  | 0.4801   | 1000    |
| bird        | 0.2757    | 0.389  | 0.3227   | 1000    |
| cat         | 0.2694    | 0.184  | 0.2187   | 1000    |
| deer        | 0.3671    | 0.185  | 0.2460   | 1000    |
| dog         | 0.3561    | 0.443  | 0.3948   | 1000    |
| frog        | 0.3695    | 0.531  | 0.4358   | 1000    |
| horse       | 0.5794    | 0.332  | 0.4221   | 1000    |
| ship        | 0.4587    | 0.494  | 0.4757   | 1000    |
| truck       | 0.4719    | 0.461  | 0.4664   | 1000    |
| **accuracy**| 0.3963    | 0.3963 | 0.3963   | 10000   |
| macro avg   | 0.4056    | 0.3963 | 0.3906   | 10000   |
| weighted avg| 0.4056    | 0.3963 | 0.3906   | 10000   |

## Limitations

- **Model Complexity**: Both models are relatively simple. Logistic regression, in particular, may be insufficient for capturing the complexities of image data in the CIFAR-10 dataset.
- **Feature Representation**: The preprocessing step converts images to grayscale and flattens them, potentially losing important spatial information.
- **Hyperparameter Tuning**: The models may not be optimally tuned, affecting their performance.

## Future Steps

- **Implement CNNs**: Convolutional Neural Networks (CNNs) are more suited for image data and could significantly improve performance.
- **Hyperparameter Optimization**: Experiment with different hyperparameters such as learning rate, batch size, and number of layers/neurons in the MLP.
- **Advanced Data Augmentation**: Use data augmentation techniques like rotation, flipping, and zooming to improve model robustness and generalization.
- **Preserve Color Information**: Instead of converting to grayscale, use the color channels to preserve more information.
- **Use Transfer Learning**: Employ pre-trained models and fine-tune them on the CIFAR-10 dataset for better performance.

These steps may improve model performance and achieve better classification accuracy on the CIFAR-10 dataset.


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