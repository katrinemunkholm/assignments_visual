
# Importing packages
import os
import sys
import cv2
import numpy as np
from numpy.linalg import norm
import pandas as pd
import argparse
from tqdm import tqdm
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append(os.path.join(".."))

# Defining argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Find similar images to the target image.")
    parser.add_argument("target_image_name", type=str, help="Name of the target image file")
    parser.add_argument("--folder_path", type=str, default=os.path.join("in", "flowers"), help="Path to the folder containing images")
    parser.add_argument("--top_n", type=int, default=5, help="Number of similar images to return")
    return parser.parse_args()

# Defining histogram class
class HistogramImageSearch:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.histograms_list = self.extract_histograms()

    def extract_histogram(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def extract_histograms(self):
        histograms_list = [(image_filename, self.extract_histogram(os.path.join(self.dataset_dir, image_filename)))
                           for image_filename in os.listdir(self.dataset_dir)]
        return histograms_list

    def compare_histograms(self, target_histogram):
        distances = [(filename, round(cv2.compareHist(target_histogram, histogram, cv2.HISTCMP_CHISQR), 2))
                     for filename, histogram in self.histograms_list]
        return distances

    def find_similar_images(self, target_image_path, num_neighbors=5):
        target_histogram = self.extract_histogram(target_image_path)
        distances = self.compare_histograms(target_histogram)
        distances = [(filename, distance) for filename, distance in distances if filename != os.path.basename(target_image_path)]
        top_n_closest = sorted(distances, key=lambda x: x[1])[:num_neighbors]
        return top_n_closest

# Defining VGG16 class
class VGG16ImageSearch:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        self.filenames = [os.path.join(self.dataset_dir, name) for name in sorted(os.listdir(self.dataset_dir))]

    def extract_features(self, img_path):
        input_shape = (224, 224, 3)
        img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img_array = img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = self.model.predict(preprocessed_img, verbose=False)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(features)
        return normalized_features

    def find_similar_images(self, target_image_path, num_neighbors=5):
        target_features = self.extract_features(target_image_path)

        feature_list = [self.extract_features(filename) for filename in tqdm(self.filenames, desc="Extracting features")]

        neighbors = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='brute', metric='cosine').fit(feature_list)

        distances, indices = neighbors.kneighbors([target_features])

        similar_images = []
        for i in range(1, num_neighbors + 1):
            similar_images.append((os.path.basename(self.filenames[indices[0][i]]), distances[0][i]))

        return similar_images

# Function to save results to CSV
def save_to_csv(results, csv_file_path):
    df = pd.DataFrame(results, columns=["Image", "Distance"])
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved to {csv_file_path}.")

# Main function
def main():
    args = parse_arguments()
    target_image_path = os.path.join(args.folder_path, args.target_image_name)

    if args.method == "histogram":
        image_search = HistogramImageSearch(args.folder_path)
        similar_images = image_search.find_similar_images(target_image_path, num_neighbors=args.top_n)
    elif args.method == "vgg":
        image_search = VGG16ImageSearch(args.folder_path)
        similar_images = image_search.find_similar_images(target_image_path, num_neighbors=args.top_n)

    output_file = os.path.join("out", "top_similar_images_test.csv")
    save_to_csv(similar_images, output_file)

if __name__ == "__main__":
    main()
