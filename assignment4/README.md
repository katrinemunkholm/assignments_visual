# Assignment 4: Historical Newspaper Face Detection

## Overview

- **Author**: Katrine Munkholm Hygebjerg-Hansen
- **Elective**: Visual Analytics, Cultural Data Science Spring 2024
- **Teacher**: Ross Deans Kristensen-McLachlan

## Description

This script analyzes historical newspaper image files to detect human faces using a pretrained CNN model (`MTCNN` from `facenet_pytorch`). It processes images from three Swiss newspapers: the Journal de Genève (JDG), the Gazette de Lausanne (GDL), and the Impartial (IMP) spanning over two centuries. The script aggregates the detection results by decade and produces both statistical summaries and visual plots representing the prevalence of human faces over time.

## Task overview

The primary goal of this project is to assess changing visual patterns in historical newspapers by detecting human faces in printed media. The specific tasks outlined for this project are:

1. **Face Detection**: Implement a face detection process using a pre-trained convolutional neural network (MTCNN) to identify human faces on newspaper pages.
2. **Data Extraction**: Analyze images from three historical Swiss newspapers: 
   - Journal de Genève (JDG, 1826-1994)
   - Gazette de Lausanne (GDL, 1804-1991)
   - Impartial (IMP, 1881-2017)
3. **Data Processing**:
   - Extract the publication year from image filenames and organize the images by decade.
   - Detect faces in each image and record the count.
4. **Data Aggregation**:
   - Group face detection results by decade.
   - Calculate and record the total number of faces and the percentage of pages featuring at least one face per decade.
5. **Output Generation**:
   - Generate CSV files summarizing the total number of faces and the percentage of pages with faces per decade for each newspaper.
   - Create plots visualizing the percentage of pages with faces over time, providing a graphical representation of trends.
6. **Analysis and Interpretation**:
   - Interpret the data to identify trends in the inclusion of human faces in newspapers over the specified period. Discuss what the trends might mean in this context.


## Data Source

The images analyzed are sourced from the Journal de Genève (JDG), the Gazette de Lausanne (GDL), and the Impartial (IMP), ranging from the early 19th to the late 20th century. Access to the images and additional documentation can be found [here](https://zenodo.org/records/3706863). For the purpose of this project, images should be downloaded and organized according to the repository structure outlined further down.


## Setup and Running the Analysis

### Requirements

- Python 3
- Libraries: 
`facenet_pytorch==2.6.0`
`matplotlib==3.8.4`
`numpy==1.26.4`
`pandas==2.2.2`
`Pillow==10.3.0`

### Environment Setup and Execution

1. **Environment Setup**:
   To set up a virtual environment and install the required packages, run the following in the command line:

   ```bash
   bash setup.sh
   ```

2. **Running the Script**:

   ```bash
   bash run.sh
   ```


## Outputs

Outputs from the script include:

- **CSV files**: Contain the total number of faces detected per decade and the percentage of pages with faces.
- **PNG plots**: Visualize the percentage of pages with faces per decade.

Outputs will be saved in the `out` directory.


## Analysis Summary and Interpretation

The plots show a significant increase in the depiction of human faces in Swiss newspapers over the studied period. Notably, the GDL and IMP showed a dramatic rise in face prevalence from the 1980s onwards. Technological advances, especially the shift from black and white to color printing in the 1980s, and a cultural shift towards visual media, may explain some of the obsrved increase in human faces in newspapers in these decades.


## Limitations and Future Work

This analysis is constrained by the quality of the newspaper scans and the performance of the face detection model. Some scans could not be processed due to their poor quality, leading to potential underestimation of faces. Future improvements could include:

### Limitations
1. **Quality of Scans**: Poor scan quality affects face detection accuracy, potentially leading to underestimations.
2. **Historical Consistency**: Changes in paper quality, print style, and photographic technology over decades may impact consistency in detection rates.


### Future Work

1. **Data Enhancement**: Use image enhancement techniques to improve the quality of poor scans before analysis.
2. **Inclusion of More Newspapers**: Broaden the study to include additional newspapers for a more comprehensive view.
3. **Cross-Verification with Other Media**: Compare trends with other media forms to validate findings and understand broader visual media trends.
4. **Machine Learning Improvements**: Train models specifically on historical imagery to adapt better to the unique characteristics of older photographs.


## Repository Structure
The script assumes the following repository structure:
```
assignment4/
│
├── in/
│   └── newspapers/
│       ├── GDL/
│           ├── <filename>.jpg
│           └── ...
│       ├── IMP/
│           ├── <filename>.jpg
│           └── ...
│       ├── JDG/
│           ├── <filename>.jpg
│           └── ...
│ 
├── out/
│   ├── GDL_faces_plot.png
│   ├── GDL_summary.csv
│   ├── IMP_faces_plot.png
│   ├── IMP_summary.csv
│   ├── JDG_faces_plot.png
│   ├── JDG_summary.csv
│ 
├── src/
│   └── script.py
│
├── requirements.txt
├── README.md
├── setup.sh
├── runLogReg.sh
└── runMLP.sh
```