"""
Assignment 1 - Simple image search algorithm
Author: Katrine Munkholm Hygebjerg-Hansen
Elective: Visual Analytics, Cultural Data Science Spring 2024
Teacher: Ross Deans Kristensen-McLachlan
"""

# including the home directory
import os
# image processing tools
import cv2
import numpy as np
# Adding parent directory to path
import sys
sys.path.append(os.path.join(".."))
# plotting tool
import matplotlib.pyplot as plt
# for handling command-line arguments
import argparse


# Function to load image from image path
def load_image(image_path):
    return cv2.imread(image_path)

# Function to calculate histogram of a given image
def calculate_histogram(image):
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError(f"The image at path '{image}' could not be loaded. Please check the file path.")
    
    # Check if the image has 3 channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("The image does not have 3 channels. Ensure it is a color image.")
    
    return cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

# Function to normalize histogram
def normalize_histogram(hist):
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

# Function to calculate the chi squared distances
def calculate_chi_squared_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

# Function to find similar images
"""
Given a folder path containing images, the function finds similar images to the target image within the folder.

Args:
    folder_path: Path to the folder containing images.
    target_image_name: Name of the target image file.
    top_n (optional): Number of similar images to return. Defaults to 5.

Returns:
   A list of containing the filenames of the top N similar images and their respective distances.
"""
def find_similar_images(folder_path, target_image_name, top_n=5):
    target_image_path = os.path.join(folder_path, target_image_name)
    target_image = load_image(target_image_path)
    target_hist = calculate_histogram(target_image)
    target_hist = normalize_histogram(target_hist)
    distances = {}
    for filename in os.listdir(folder_path):
        if filename != target_image_name and filename.endswith("jpg"):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            if image is not None and len(image.shape) == 3 and image.shape[2] == 3:  
                hist = calculate_histogram(image)
                hist = normalize_histogram(hist)
                distance = calculate_chi_squared_distance(target_hist, hist)
                distances[filename] = distance
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    return sorted_distances[:top_n]

# Function to save results 
def save_to_csv(results, csv_file_path):
    with open(csv_file_path, 'w') as f:
        f.write("Image,Distance\n")
        for item in results:
            f.write(f"{item[0]},{item[1]}\n")
    print("CSV file saved to folder 'out'.")

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Find similar images to the target image.")
    parser.add_argument("target_image_name", type=str, help="Name of the target image file")
    parser.add_argument("--folder_path", type=str, default=os.path.join("in", "flowers"), help="Path to the folder containing images")
    parser.add_argument("--top_n", type=int, default=5, help="Number of similar images to return")
    return parser.parse_args()

# Main function to run the task
def main():
    args = parse_arguments()
    top_5_similar = find_similar_images(args.folder_path, args.target_image_name, top_n=args.top_n)
    csv_file_path = os.path.join("out", "top_5_similar_images.csv")
    save_to_csv(top_5_similar, csv_file_path)


if __name__ == "__main__":
    main()
