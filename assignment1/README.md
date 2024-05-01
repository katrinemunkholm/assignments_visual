# Building a Simple Image Search Algorithm

The script 'Image_search.py' performs a simple image search from color histograms of provided images.
The algorithm compares the color histograms of a specified target image to those of other images from the provided dataset. Finally, it saves a .CSV file in the "out" folder containing names of the images that are most similar to the target image. Deafult top n of images is set to 5.

## Data source
The data is a 17 category flower dataset with 80 images for each class. The flowers chosen are some common flowers in the UK. The images have large scale, pose and light variations and there are also classes with large varations of images within the class and close similarity to other classes.
[Data source](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

The data set is included in the repository in the folder "data/flowers". Otherwise, it can be downloaded [Here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz).


## Requirements
- Python 3
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pandas (`pandas`)

## Reproduction
1. Clone the repository: git clone <https://github.com/katrinemunkholm/cds_visual_assignments.git>

2. The data set should be named "flowers" and placed in the folder "data"

3. Install the required dependencies: pip install -r requirements.txt

4. Run the script: python Image_search.py

5. Check the "out" folder for the CSV file containing the top 5 similar images to the target image.



## Output
The script saves the results to a .CSV file in the output folder. The CSV file contains the filenames of the top 5 images that are most similar to the target image along with their respective Chi-Squared distance to the target image.


## Results and summary

In a test run, image_0158.jpg was chosen chosen as target image. When running the script, the algorithm was able to provide the filenames of the 5 images whose histograms had the lowest chi-squared distance to that of the target image, with the lowest value being 37.81 for image_0393.jpg.
The differences in distances indicate varying degrees of similarity, with closer distances implying higher similarity and vice versa. However, when comparing the images in the dataset, it's important to note that the provided algorithm relies solely on color information and may easily overlook other visual feautures which could be important for determining similarity. An example of this could be shape similarity of the flowers in the images.


## File Structure
The project directory is structured as follows:

```

assignment1/
│
├── data/
│   └── flowers/
│       ├── image_0001.jpg
│       ├── image_0002.jpg
│       └── ...
└── out/
    └── top_5_similar_images.csv 
│
├── src/
│   └── Image_search.py
│   
├── README.md
└── requirements.txt

```