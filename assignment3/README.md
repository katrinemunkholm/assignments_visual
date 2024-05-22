# Assignment 3 - Document Classification using Pretrained Image Embeddings

## Description

This repository contains Python code developed to classify documents based on their visual appearance using transfer learning with pretrained Convolutional Neural Networks (CNNs), specifically leveraging the VGG16 architecture. The task demonstrates the ability to predict document types from their visual characteristics without analyzing the text content. The project utilizes the Tobacco3482 dataset, showcasing a variety of document types.


### Task Overview
For this assignment, the main objective was to explore whether visual features extracted from document images could be effectively used to classify different types of documents. The specific tasks to be accomplished include:

1. Load and organize the Tobacco3482 dataset into an appropriate structure for processing.
2. Utilize a pretrained VGG16 model to extract visual features from each document image.
3. Train a classifier that uses these features to predict the document's type.
4. Generate a classification report detailing the accuracy, precision, recall, and F1-score for each document type.
5. Produce learning curves to visualize the model's training and validation accuracy, as well as loss over time.
6. Discuss the model's performance, identify limitations, and suggest possible improvements.


## Data Source

The dataset used is the 'Tobacco3482', which is a subset of a larger corpus detailed in its [original paper](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). This subset includes 3,842 images across 10 different document types such as advertisements, emails, forms, and more, reflecting a diverse set of document appearances.

- **Access**: Download the dataset [here.](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg/download?datasetVersionNumber=1)
For local setup, the dataset should be downloaded and organized in the directory structure expected by the script, which can be found further down.


## Setup and Running the Analysis
### Requirements

- Python 3
- Libraries: 
`matplotlib==3.8.4`
`numpy==1.26.4`
`scikit_learn==1.4.2`
`tensorflow==2.16.1`


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

The script will generate:
- A **classification report** saved in the output directory, detailing the precision, recall, and F1-score for each document type.
- **Learning curves** for the training process, saved as a PNG file in the output directory. These plots provide insights into the loss and accuracy metrics throughout the epochs.

## Speficications for fone-tuning

Different specifications have been implemented in the `build_model()` function to fine-tune the CNN architecture based on the VGG16 model, aiming to enhance performance in document classification tasks. 

These include:
- **Feature Extraction Enhancement**: Added a fully connected layer with 256 units and ReLU activation function to improve feature extraction capabilities.
- **Stability Improvement**: Applied batch normalization to stabilize training and improve convergence speed by normalizing the input to each layer.
- **Prevent Overfitting**: Implemented dropout regularization with a dropout rate of 0.5 to prevent overfitting by randomly dropping neurons during training.
- **Fine-Tuning Strategy**: Unfroze the last convolutional block of the pre-trained VGG16 model by allowing the last 4 layers to be trainable, enabling further adaptation to the specific document classification task.
- **Optimization Strategy**: Compiled the model with Stochastic Gradient Descent (SGD) optimizer, employing a lower learning rate of 0.0001 and momentum of 0.9 for fine-tuning, aiming for controlled updates and stable convergence during training.

These modifications are aimed at enhancing the model's ability to extract discriminative features from document images and improve classification accuracy. Gradually implementing these did indeed improve model performance, resulting in the final classification report and learning curve which can be found in the folder 'out'.

## Results and Summary

The project results in a trained classifier capable of distinguishing between various types of documents based on their visual layout. Key findings include:
- High accuracy for visually distinct document types such as advertisements and emails.
- Lower performance on more heterogeneously styled documents like scientific papers, which can vary significantly in appearance.

### Classification report

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADVE         | 0.82      | 0.88   | 0.85     | 57      |
| Email        | 0.93      | 0.94   | 0.94     | 135     |
| Form         | 0.75      | 0.83   | 0.79     | 88      |
| Letter       | 0.76      | 0.82   | 0.79     | 122     |
| Memo         | 0.67      | 0.83   | 0.74     | 109     |
| News         | 0.72      | 0.68   | 0.70     | 34      |
| Note         | 0.85      | 0.61   | 0.71     | 36      |
| Report       | 0.72      | 0.48   | 0.57     | 48      |
| Resume       | 0.67      | 0.67   | 0.67     | 15      |
| Scientific   | 0.64      | 0.40   | 0.49     | 53      |
| **accuracy** |           |        | 0.77     | 697     |
| macro avg    | 0.75      | 0.71   | 0.72     | 697     |
| weighted avg | 0.77      | 0.77   | 0.77     | 697     |


The classification report reveals high F1-scores across multiple classes, particularly notable in classes such as "Email" and "Note," where predictions were accurate approximately 93% and 85% of the time, respectively.
Instances of lower accuracy scores are evident in the "Scientific" class. One possible explanation for this could be the diverse nature of scientific disciplines, each adhering to distinct conventions, resulting in varying visual appearances of articles and papers. This diversity may pose challenges for the CNN in accurately classifying documents belonging to this category. Overall, the model demonstrates satisfactory performance in classifying documents across the 10 categories solely based on their visual appearance, as indicated by the weighted average f1-score of 0.77.

When visually inspecting the loss curves, they indicate a model which generally has a good fit to the data. 
Both train_loss and val_loss are decreasing gradually through the epochs, indicating that they are learning well from the data. Though there is a slight gap between the two curves, the distance hereof is rather consistent, indicating that the model is not just overfitting to the training data. Additionally, the curves seem to be flattening out at the end, indicating that the chosen number of epochs is good fit for this model and provided data.



## Limitations and steps for improvement

### Limitations
- The model's performance is limited by the visual diversity within some document categories.
- Dependence on the quality and representativeness of the pre-trained embeddings.

### Improvement Steps
- Larger dataset to include a wider variety of document appearances.
- Experimenting with different architectures and more advanced transfer learning techniques.
- Implementing more robust data augmentation strategies to improve generalization.

## File structure

The script assumes the following file structure:
```
assignment3/
│
├── in/
│   └── Tobacco3482/
│        ├── ADVE/
│        │   ├── <filename>.jpg
│        │   └── ...
│        ├── Email/
│        │   ├── <filename>.jpg
│        │   └── ...
│        └── ...
│
├── out/
│   ├── classification_report.txt
│   └── learning_curves.png
├── src/
│   └── script.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```