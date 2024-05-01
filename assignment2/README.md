# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## Description

These scripts are designed to solve the task of classifying the CIFAR-10 dataset using two different approaches: logistic regression and neural networks. The goal is to build reproducible pipelines for a machine learning approach and demonstrate the usage of scikit-learn for building benchmark classifiers on image classification data. 
For the CIFAR-10 dataset, these classifiers classify the images into 10 different categories, namely 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship' and 'truck'.

The output of the project includes:
1. Classification reports on performance for both logistic regression and neural network classifiers.
2. Plot of the loss curve during training for the neural network classifier.

The purpose of these scripts is to showcase the process of loading and preprocessing the data, training the classifiers, evaluating model performance, and saving results in a structured manner. By running these scripts, the user can build simple benchmark classifiers for image classification tasks. 

'out' contains:
- Classification reports for both models (Logistic regression classifier and MLP classifier)
- The loss curve of the neural network classifier 

'src' contains:
- Two scripts, one for each classification model

## Data source

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

The data set is in the provided repository in the folder 'data'. 
Otherwise, you can download the CIFAR-10 python version dataset [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)


## Reproduction
1. Clone the repository: git clone <https://github.com/katrinemunkholm/cds_visual_assignments.git>

2. The data set should be named "cifar10" and placed in the folder "data"

3. Install the required dependencies: pip install -r requirements.txt

4. Run the scripts: python Logistic_reg_classifier.py and python Neural_network_classifier.py

5. Check the 'out' folder for the two classification reports and the loss curve of the MLP classifier.


## Results and summary

The three output files are saved in the folder 'out'. 

The evaluation of the Logistic Regression model reveals a relatively low performance, with a weighted average f1-score slightly exceeding 0.31. In comparison, the MLP demonstrates just slightly better performance, with a weighted average f1-score slightly above 0.39.
The loss curve of the MLP is gradually decreasing as the model is trained. This indicates that the model is learning through the number of iterations. 

An alternative recommendation for models with potentially higher classification capabilities includes Convolutional Neural Networks (CNNs).
Assignment 3 provides an example of implementing a Convolutional Neural Network (CNN) on a different data set.
