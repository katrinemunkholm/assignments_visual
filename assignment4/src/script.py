"""
Assignment 4 - Historical Newspaper Face Detection
Author: Katrine Munkholm Hygebjerg-Hansen
Elective: Visual Analytics, Cultural Data Science Spring 2024
Teacher: Ross Deans Kristensen-McLachlan
"""

# Import neccessary libraries. Face detection, data manipulation and numerical operations
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

def detect_faces(image_path):
    """Detect faces in an image using MTCNN and return the count."""
    try:
        img = Image.open(image_path)
        boxes, _ = mtcnn.detect(img)
        face_count = 0 if boxes is None else len(boxes)
        return face_count
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def extract_year_from_filename(filename):
    """Extract the year from the filename assuming format 'XXX-YYYY-MM-DD-etc.jpg'."""
    match = re.search(r'-([0-9]{4})-', filename)
    if match:
        return int(match.group(1))
    return None

def analyze_newspaper_folder(folder_path):
    """Analyze all images in a newspaper folder and summarize data by decade."""
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        year = extract_year_from_filename(filename)
        if year:
            face_count = detect_faces(file_path)
            data.append((year, face_count, face_count > 0))
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Year', 'FaceCount', 'HasFaces'])
    
    # Group by decade
    df['Decade'] = (df['Year'] // 10) * 10
    summary = df.groupby('Decade').agg(
        TotalFaces=('FaceCount', 'sum'),
        PagesWithFaces=('HasFaces', 'sum'),
        TotalPages=('Year', 'count')
    )
    summary['PercentageWithFaces'] = (summary['PagesWithFaces'] / summary['TotalPages']) * 100
    return summary

def generate_plots_and_csv(data, output_dir, newspaper_name):
    """Generate plots and CSV files from summarized data."""
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, f'{newspaper_name}_summary.csv')
    data.to_csv(csv_file)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['PercentageWithFaces'], marker='o', linestyle='-')
    plt.title(f'Percentage of Pages with Faces by Decade: {newspaper_name}')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Pages with Faces')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{newspaper_name}_faces_plot.png'))
    plt.close()

def main():
    # Paths to newspaper folders
    input_dir = "in/newspapers"
    output_dir = "out"
    newspapers = ["GDL", "IMP", "JDG"]
    
    for newspaper in newspapers:
        folder_path = os.path.join(input_dir, newspaper)
        summarized_data = analyze_newspaper_folder(folder_path)
        generate_plots_and_csv(summarized_data, output_dir, newspaper)

if __name__ == '__main__':
    main()

