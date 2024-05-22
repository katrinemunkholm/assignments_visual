# Assignment 4: Historical Newspaper Face Detection

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
`Pillow==10.2.0`

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


### GDL summary, rounded

| Decade | Total Faces | Pages With Faces | Total Pages | Percentage With Faces |
|--------|-------------|------------------|-------------|-----------------------|
| 1790   | 3           | 3                | 20          | 15.00                 |
| 1800   | 10          | 9                | 36          | 25.00                 |
| 1810   | 2           | 2                | 26          | 7.69                  |
| 1820   | 1           | 1                | 32          | 3.13                  |
| 1830   | 2           | 2                | 36          | 5.56                  |
| 1840   | 0           | 0                | 24          | 0.00                  |
| 1850   | 2           | 2                | 24          | 8.33                  |
| 1860   | 1           | 1                | 24          | 4.17                  |
| 1870   | 1           | 1                | 24          | 4.17                  |
| 1880   | 1           | 1                | 24          | 4.17                  |
| 1890   | 1           | 1                | 26          | 3.85                  |
| 1900   | 4           | 4                | 28          | 14.29                 |
| 1910   | 4           | 4                | 30          | 13.33                 |
| 1920   | 8           | 6                | 28          | 21.43                 |
| 1930   | 8           | 6                | 28          | 21.43                 |
| 1940   | 10          | 8                | 38          | 21.05                 |
| 1950   | 8           | 8                | 48          | 16.67                 |
| 1960   | 25          | 19               | 90          | 21.11                 |
| 1970   | 14          | 11               | 78          | 14.10                 |
| 1980   | 60          | 44               | 136         | 32.35                 |
| 1990   | 115         | 74               | 208         | 35.58                 |

### IMP Summary, rounded

| Decade | Total Faces | Pages With Faces | Total Pages | Percentage With Faces |
|--------|-------------|------------------|-------------|-----------------------|
| 1880   | 1           | 1                | 34          | 2.94                  |
| 1890   | 17          | 12               | 52          | 23.08                 |
| 1900   | 25          | 17               | 68          | 25.00                 |
| 1910   | 27          | 17               | 52          | 32.69                 |
| 1920   | 46          | 29               | 64          | 45.31                 |
| 1930   | 31          | 19               | 54          | 35.19                 |
| 1940   | 24          | 16               | 56          | 28.57                 |
| 1950   | 105         | 47               | 94          | 50.00                 |
| 1960   | 175         | 60               | 144         | 41.67                 |
| 1970   | 202         | 82               | 200         | 41.00                 |
| 1980   | 343         | 106              | 210         | 50.48                 |
| 1990   | 210         | 96               | 202         | 47.52                 |
| 2000   | 657         | 153              | 216         | 70.83                 |
| 2010   | 699         | 146              | 188         | 77.66                 |



### JDG Summary, rounded

| Decade | Total Faces | Pages With Faces | Total Pages | Percentage With Faces |
|--------|-------------|------------------|-------------|-----------------------|
| 1820   | 0           | 0                | 26          | 0.00                  |
| 1830   | 1           | 1                | 54          | 1.85                  |
| 1840   | 1           | 1                | 50          | 2.00                  |
| 1850   | 2           | 2                | 48          | 4.17                  |
| 1860   | 4           | 3                | 48          | 6.25                  |
| 1870   | 5           | 2                | 46          | 4.35                  |
| 1880   | 6           | 5                | 50          | 10.00                 |
| 1890   | 1           | 1                | 52          | 1.92                  |
| 1900   | 11          | 9                | 54          | 16.67                 |
| 1910   | 6           | 4                | 50          | 8.00                  |
| 1920   | 21          | 11               | 110         | 10.00                 |
| 1930   | 32          | 20               | 104         | 19.23                 |
| 1940   | 13          | 10               | 92          | 10.87                 |
| 1950   | 49          | 29               | 116         | 25.00                 |
| 1960   | 44          | 31               | 166         | 18.67                 |
| 1970   | 48          | 41               | 186         | 22.04                 |
| 1980   | 143         | 106              | 306         | 34.64                 |
| 1990   | 190         | 123              | 424         | 29.01                 |

### Trends and Interpretation

- **Increase in Visual Content**: Across all three newspapers, there is a noticeable rise in the number of faces and the percentage of pages with faces towards the more recent decades, indicating a growing emphasis on visual content.

- **Variability Among Newspapers**: The overall trend is consistent, but the extent and timeline vary. The IMP shows a dramatic increase from the 1890s to the 2000s, peaking at 77.66% in the 2010s. In contrast, the GDL has a more gradual increase with significant jumps in the 1980s and 1990s.

- **Historical Context**: The presence of faces on newspaper pages is linked to technological advancements and cultural shifts. Limited visual content in the early 19th century was due to technological constraints. As printing technology, photography, and digital imaging improved, newspapers began incorporating more images. Sharp increases in the mid to late 20th century align with advanced printing techniques and a cultural shift towards more visually engaging media. This context aligns with the observed results.

- **Reader Preferences and Competition**: The rise of television and digital media, which are inherently visual, likely pressured newspapers to enhance their visual content to remain competitive.

Overall, the increasing presence of faces and visual content in newspapers underscores a transformation in how information is presented, reflecting broader societal and technological changes over the centuries.


## Limitations and Future Work

This analysis is influenced by the quality of the newspaper scans and the performance of the face detection model. Several scans could not be processed due to poor quality, which may result in an underestimation of the number of faces. Specifically, 3 images from GDL, 2 images from IMP, and 1 image from JDG were not processed, as indicated by terminal messages during script execution. Future improvements could include:

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
└── run.sh
```