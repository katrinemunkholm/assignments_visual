"""
Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks, script #2 out of #2
Author: Katrine Munkholm Hygebjerg-Hansen
Elective: Visual Analytics, Cultural Data Science Spring 2024
Teacher: Ross Deans Kristensen-McLachlan
"""

## Importing necessary libraries and appending parent directory to system path
import os
import sys
sys.path.append("..")

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util
import tensorflow as tf
import keras as keras

# Import sklearn metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Importing matplotlib for data visualization (loss curve)
import matplotlib.pyplot as plt
# Importing pandas to create dataframe
import pandas as pd

#Load the data and check its shape
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    return x_train, y_train, x_test, y_test


# Define class names of categories
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Defining function to convert images to grayscale
def rgb_to_grayscale(images):
    grayscale_images = np.mean(images, axis=-1)
    return grayscale_images

#Preprocessing of the data
def preprocess_data(x_train, x_test):
    x_train_grayscale = rgb_to_grayscale(x_train)
    x_test_grayscale = rgb_to_grayscale(x_test)
    x_train_scaled = x_train_grayscale / 255.0
    x_test_scaled = x_test_grayscale / 255.0
    x_train_reshaped = x_train_scaled.reshape(-1, 1024)
    x_test_reshaped = x_test_scaled.reshape(-1, 1024)
    return x_train_reshaped, x_test_reshaped

# Create and train the MLP classifier
def train_mlp_classifier(x_train_reshaped, y_train):
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=1)
    mlp_clf.fit(x_train_reshaped, y_train.ravel())
    return mlp_clf

# Save the loss curve plot
def save_loss_curve(mlp_clf):
    plt.figure()
    plt.plot(mlp_clf.loss_curve_)
    plt.title('Loss Curve During Training')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join("out/loss_curve_MLP.png"))

# Create metrics for classification report, create report and save it
def evaluate_mlp_classifier(mlp_clf, x_test_reshaped, y_test, class_names):
    mlp_predictions = mlp_clf.predict(x_test_reshaped)
    mlp_accuracy = accuracy_score(y_test, mlp_predictions)
    print("MLP Accuracy:", mlp_accuracy)
    mlp_report = classification_report(y_test, mlp_predictions, target_names=class_names)
    print("MLP Classification Report:")
    print(mlp_report)
    mlp_classification_report = pd.DataFrame(metrics.classification_report(y_test, mlp_predictions, target_names=class_names, output_dict=True)).transpose()
    mlp_classification_report.to_csv(os.path.join("out/MLP_classification_report.csv"))
    print("A plot of the loss curve during training has been saved as a .png file in the folder 'out'")

#Main function for running the script
def main():
    x_train, y_train, x_test, y_test = load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    x_train_reshaped, x_test_reshaped = preprocess_data(x_train, x_test)
    mlp_clf = train_mlp_classifier(x_train_reshaped, y_train)
    save_loss_curve(mlp_clf)
    evaluate_mlp_classifier(mlp_clf, x_test_reshaped, y_test, class_names)

# Running the main()
if __name__ == "__main__":
    main()
