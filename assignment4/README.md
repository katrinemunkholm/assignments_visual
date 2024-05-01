# Assignemnt 4: Historical Newspaper Face Detection

- Author: Katrine Munkholm Hygebjerg-Hansen
- Elective: Visual Analytics, Cultural Data Science Spring 2024
- Teacher: Ross Deans Kristensen-McLachlan

## Description

This script analyzes historical newspaper image files to detect human faces using a pretrained CNN model (`MTCNN` from `facenet_pytorch`). It processes images from three Swiss newspapers: the Journal de Genève (JDG), the Gazette de Lausanne (GDL), and the Impartial (IMP) spanning over two centuries. The script aggregates the detection results by decade and produces both statistical summaries and visual plots representing the prevalence of human faces over time.

## Data

The data can be accessed for download [here](https://zenodo.org/records/3706863) and shold be placed in the "in" folder. See assumed repo struture below.


## Features

- **Face Detection:** Utilizes the `MTCNN` model to detect faces within images of newspaper pages.
- **Data Aggregation by Decade:** Organizes the results to show the count of faces and the percentage of pages with at least one face detected, grouped by decade.
- **Output Generation:** Produces CSV files for detailed statistics and PNG files for visual representation of trends over the decades.

## Reproduction

The script assumes the following structure of the repository:
```
historical_newspapers_analysis/
│
├── in/
│   └── newspapers/
│       ├── GDL/
│       ├── IMP/
│       ├── JDG/
│ 
├── out/
│     ├── GDL_faces_plot.png
│     ├── GDL_summary.csv
│     ├── IMP_faces_plot.png
│     ├── IMP_summary.csv
│     ├── JDG_faces_plot.png
│     ├── JDG_summary.csv
│ 
├── src/
│ └── script.py
│
├── requirements.txt
└── README.md
```

1. Install required packages by running the following code in the command line:

```bash
pip install -r requirements.txt
```

2. To run the script, use the following command:

```bash
python src/script.py
```

## Output

After running the script, the following output is generated:

- CSV files detailing the total number of faces detected and the percentage of pages with faces per decade for each newspaper.
- Plots visualizing the percentage of pages with faces by decade, saved as PNG files.

Example Output:
- Files like `GDL_faces_plot.png` and `GDL_summary.csv` in the `out/` directory show the results for the Gazette de Lausanne newspaper.


## Summary and intepretation

When inspecting the plots, a clear tendency of an increase in faces on newspaper pages is observed. The increase is gradual over time for all three of the newspapers, however, for IMP, for which the most recent data in the data set is avaliable, a big increase is seen, ranging from around 3% in 1880's all the way up to approx. 78% in 2010's. From the 1990's to the 2010's an increase of 20% is observed on the plot.

Error handling was neccessary to implement in the code, as some of the scans were not possible to analyze due to truncation. This naturally disrupts the analysis a bit, though this was only seen for a handful og files across the three newspapers.
