"""
Assignment 1 - Simple image search algorithm, CNN
Author: Katrine Munkholm Hygebjerg-Hansen
Elective: Visual Analytics, Cultural Data Science Spring 2024
Teacher: Ross Deans Kristensen-McLachlan
"""

# including the home directory
import os
# image processing tools
import cv2
import numpy as np 
from numpy.linalg import norm  
# For loading and converting images
from tensorflow.keras.preprocessing.image import load_img, img_to_array  
# VGG16 model and preprocessing
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  
 # For k-nearest neighbors functionality
from sklearn.neighbors import NearestNeighbors 
# For writing CSV files
import csv
# For argument parsing
import argparse

def extract_features(img_path, model):
    """
    Extract features from an image using a specified model.
    
    Args:
    img_path (str): Path to the image file.
    model (Model): Preloaded TensorFlow model used for feature extraction.
    
    Returns:
    numpy.ndarray: Normalized features extracted from the image.
    """
    input_shape = (224, 224, 3)
    img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=0)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

def load_model():
    """
    Load and return the VGG16 model pre-trained on ImageNet.
    
    Returns:
    Model: The VGG16 model with pre-trained weights.
    """
    return VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def image_search(data_dir, target_img_path, top_n=5):
    """
    Perform image search to find top_n similar images to a target image in a specified directory.
    
    Args:
    data_dir (str): Directory containing images to search through.
    target_img_path (str): Path to the target image.
    top_n (int): Number of top similar images to find.
    
    Returns:
    list: List of tuples with filename and distance of the top_n similar images.
    """
    model = load_model()
    target_features = extract_features(target_img_path, model)
    feature_list = []
    filenames = []
    for filename in sorted(os.listdir(data_dir)):
        if os.path.join(data_dir, filename) != target_img_path and filename.endswith("jpg"):  # Exclude target image and ensure file is a .jpg
            file_path = os.path.join(data_dir, filename)
            features = extract_features(file_path, model)
            feature_list.append(features)
            filenames.append(filename) 
    neighbors = NearestNeighbors(n_neighbors=top_n, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([target_features])
    nearest_images = [(filenames[idx], dist) for dist, idx in zip(distances[0], indices[0])]
    return nearest_images

def save_results(results, output_dir):
    """
    Save the search results to a CSV file in the specified directory.
    
    Args:
    results (list): List of tuples containing filenames and distances of similar images.
    output_dir (str): Directory where the results CSV will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'top_5_similar_images_CNN.csv')
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Distance'])
        for image, distance in results:
            writer.writerow([image, distance])

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Find similar images to the target image.")
    parser.add_argument("target_image_name", type=str, help="Name of the target image file")
    parser.add_argument("--folder_path", type=str, default=os.path.join("in", "flowers"), help="Path to the folder containing images")
    parser.add_argument("--top_n", type=int, default=5, help="Number of similar images to return")
    return parser.parse_args()

# Main function to run tasks
def main():
    args = parse_arguments()
    data_directory = args.folder_path
    target_image_path = os.path.join(data_directory, args.target_image_name)
    top_n = args.top_n
    output_directory = 'out'
    
    results = image_search(data_directory, target_image_path, top_n)
    save_results(results, output_directory)
    print(f'Results saved to {output_directory}/top_5_similar_images_CNN.csv')

if __name__ == "__main__":
    main()
