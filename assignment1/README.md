# Assignment 1: Building a simple image search algorithm

## Overview

- **Author**: Katrine Munkholm Hygebjerg-Hansen
- **Elective**: Visual Analytics, Cultural Data Science Spring 2024
- **Teacher**: Ross Deans Kristensen-McLachlan

## Description

This repository contains scripts for image search algorithms using color histograms and convolutional neural networks (CNN). The algorithms are designed to compare a target image against a data set of images to find the most visually similar ones based on color distribution and deep learning features.


## Task Overview
The purpose of this project was to create an image search algorithm that can find similar images from a collection of over a thousand flower photographs. The specific tasks accomplished in this project included:

1. Identifying a particular image to use as the benchmark.
2. Using OpenCV to extract the color histogram of the benchmark image.
3. Generating color histograms for the entire image collection in the dataset.
4. Analyzing these histograms by applying OpenCV's `cv2.compareHist()` with the `cv2.HISTCMP_CHISQR` comparison method.
5. Developing a secondary approach utilizing a pre-trained CNN (specifically VGG16) and K-Nearest Neighbors to extract and compare image features.
6. Identifying and documenting the five most similar images to the benchmark for both the histogram and CNN-based methods.
7. Compiling the results into a CSV file, saved in the 'out' folder, with columns indicating the Filename and Distance.


## Data Source

The data set used is the "17 category flower dataset" from the Visual Geometry Group at the University of Oxford. It features 17 different flower species, with 80 images per species, showing variations in scale, pose, and lighting.

- **Direct download**: [17flowers.tgz](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz)
- **More information**: [Dataset details](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

## File Structure

The downloaded folder should be placed in the "in" folder and named "flowers". 
The script assumes for following structure of the repository:

```
assignment1/
│
├── in/
│   └── flowers/
│       ├── image_0001.jpg
│       ├── image_0002.jpg
│       └── ...
├── out/
│    └── top_5_similar_images.csv 
│    └── top_5_similar_images_CNN.csv 
│
├── src/
│   └── Image_search.py
│   └── Image_search_cnn.py
│   
├── README.md
├── requirements.txt
├── setup.sh
├── run.sh
└── runCNN.sh

```

## Command-Line Arguments
This script accepts the following command-line arguments to configure its behavior. The arguments can be specified when running the script from the command line.

### Required Argument
- `target_image_name` (str): The name of the target image file for which similar images need to be found. This argument is required.
  - Example: `image_0158.jpg`

### Optional Arguments
- `--folder_path` (str): The path to the folder containing the images to search through. Defaults to `'in/flowers'` if not specified.
  - Example: `--folder_path path/to/your/folder`
- `--top_n` (int): The number of top similar images to return. Defaults to `5` if not specified.
  - Example: `--top_n 10`


## Setup and Running the Analysis

### Requirements

- Python 3
- Libraries: 
`matplotlib==3.8.4`
`numpy==1.26.4`
`pandas==2.2.2`
`scikit_learn==1.4.2`
`seaborn==0.13.2`
`tensorflow==2.16.1`
`tqdm==4.66.4`
`opencv-python==4.9.0.80`

### Environment Setup and Execution

1. **Environment Setup**:
   To set up a virtual environment and install the required packages, run the following in the command line:
   ```bash
   bash setup.sh
   ```

2. **Running the Scripts**:
   - For histogram-based image search:
     ```bash
     bash run.sh `target_image_name`
     ```
   - For CNN-based image search:
     ```bash
     bash runCNN.sh `target_image_name`
     ```
#### Example: Running the script with a specified target image

```bash
bash run.sh image_0158.jpg
```

## Output

Results are saved to `.CSV` files within the "out" directory. Each CSV file lists the top similar images to the target image, along with their respective distances. Example output files show the effectiveness of both methods in identifying similar images.

## Results and Summary

In the tests using the default settings, the algorithms successfully identified images with color and feature patterns most similar to those of the target image. The results include distances that quantify the similarity, providing a clear metric to evaluate the outcome.

In a test run, image_0158.jpg was chosen chosen as target image. When running the scripts, the algorithm was able to provide the filenames of the 5 images whose histograms had the lowest chi-squared distance to that of the target image, with the lowest value being 37.81 for image_0393.jpg.
The differences in distances indicate varying degrees of similarity, with closer distances implying higher similarity and vice versa. An example of this could be shape similarity of the flowers in the images.



## Limitations and Future Improvements

The current algorithms, while effective, rely heavily on specific features (color histograms for the first method and deep features for the CNN method). 

When comparing the images in the dataset, it's important to note that the provided algorithm relies solely on color information for the first method and deep features for the CNN method, which is why they may easily overlook other visual feautures which could be important for determining similarity.
Current limitations and future improvements could include:

### Current Limitations

#### Color Histogram-Based Method
- **Lacks Structural Information**: Only captures color distribution, missing spatial arrangement.

#### CNN-Based Method
- **Computationally Intensive**: Requires significant processing power, especially with large datasets.
- **Limited by Pre-trained Weights**: Performance depends on the relevance of pre-trained weights from ImageNet.
- **Overlooks Some Features**: May miss fine-grained textures or specific shapes.

### Future Improvements

- **Additional Features**: Integrate texture and shape features for more comprehensive similarity measures.
- **Hybrid Approaches**: Combine color, texture, shape, and deep features for balanced comparisons.
- **Domain-Specific Fine-Tuning**: Fine-tune CNN models on domain-specific datasets.
- **User Feedback**: Implement feedback mechanisms to refine search results based on user input.

By addressing these areas, the image search algorithms might become more accurate and versatile.