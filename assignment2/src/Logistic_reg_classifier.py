"""
Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks, script #1 out of #2
Author: Katrine Munkholm Hygebjerg-Hansen
Elective: Visual Analytics, Cultural Data Science Spring 2024
Teacher: Ross Deans Kristensen-McLachlan
"""

# Importing libraries and setting path
import os
import sys
import pandas as pd
sys.path.append("..")

#Import tensorflow for dataset
import tensorflow as tf
import keras as keras

#Import matplot for plots
import matplotlib.pyplot as plt

#Import numpy
import numpy as np

#Import Sklearn metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics

#Import teaching utils
import utils.classifier_utils as clf_util

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

# Converting images to grayscale
def rgb_to_grayscale(images):
    grayscale_images = np.mean(images, axis=-1)
    return grayscale_images

# Preprocessing of the data. 
''' 
Grayscaling, scaling and reshaping. 
Returns the preprocessed data.

'''
def preprocess_data(x_train, x_test):
    x_train_grayscale = rgb_to_grayscale(x_train)
    x_test_grayscale = rgb_to_grayscale(x_test)
    x_train_scaled = x_train_grayscale / 255.0
    x_test_scaled = x_test_grayscale / 255.0
    x_train_reshaped = x_train_scaled.reshape(-1, 1024)
    x_test_reshaped = x_test_scaled.reshape(-1, 1024)
    return x_train_reshaped, x_test_reshaped

# Training logistic regression classifier
def train_logistic_regression(x_train_reshaped, y_train):
    clf = LogisticRegression(tol=0.1, solver='saga', multi_class='multinomial')
    clf.fit(x_train_reshaped.reshape((x_train_reshaped.shape[0], -1)), y_train.ravel())
    return clf

# Plot and save results 
def plot_results(coefs, x_test_reshaped, y_test, clf, class_names):
    clf_util.plot_coefs(coefs, len(class_names))
    clf_util.plot_individual(x_test_reshaped.reshape((x_test_reshaped.shape[0], -1)), y_test, 50)
    clf_util.plot_probs(x_test_reshaped, 50, clf, class_names)
    y_pred = clf.predict(x_test_reshaped)
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of predictions on test set is:", accuracy)
    cm = metrics.classification_report(y_test, y_pred, target_names=class_names)
    print("The metrics of the classification report are:", cm)
    clf_util.plot_cm(y_test, y_pred, normalized=True)
    report_df = pd.DataFrame(metrics.classification_report(y_test, y_pred, target_names=class_names, output_dict=True)).transpose()
    report_df.to_csv("../out/Logistic_regression_classification_report.csv")
    print("The classification report is saved as 'Logistic_regression_classification_report.CSV' in the folder 'out'")


# Main function to perform tasks
def main():
    x_train, y_train, x_test, y_test = load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    x_train_reshaped, x_test_reshaped = preprocess_data(x_train, x_test)
    clf = train_logistic_regression(x_train_reshaped, y_train)
    print("Shape of the coefficient (weights) matrix:", clf.coef_.shape)
    plot_results(clf.coef_, x_test_reshaped, y_test, clf, class_names)

# Running the main ()
if __name__ == "__main__":
    main()
