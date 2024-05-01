# Assignment 3 - Transfer learning w/ pretrained CNNs

## Description

This repository contains code developed as part of Assignment 3 for the course Visual Analytics. The script provided utilizes convolutional neural networks (CNNs) to perform document classification based on visual appearance of documents rather than their actual content. The main functionality of the script involves loading a dataset of document images (Tobacco3482 is used), extracting pretrained image embeddings using a VGG16 CNN architecture, and training a classifier to predict document types from the defined label names. The classifier is then evaluated using classification metrics, and learning curves are generated to visualize the model's performance during training. 


## Data 

For this classification, the dataset 'Tobacco3482' is used. 
The dataset comprises a collection of 3,482 document images categorized into 10 distinct types. Each document type represents a different category, including advertisements (ADVE), emails, forms, letters, memos, news articles, notes, reports, resumes, and scientific papers. These documents exhibit a variety of visual appearances and content, reflecting real-world document diversity. 
You can download and read more about the dataset [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download)


## Reproduction
1. Clone the repository: git clone <https://github.com/katrinemunkholm/cds_visual_assignments.git>

2. The data set should be named "Tobacco3482" and placed in the folder "data"

3. Install the required dependencies: pip install -r requirements.txt

4. Run the script: python script.py

5. Check the "out" folder for the classification report and the loss curve of the CNN. 


## Output

When running the script, a classification report containing metrics on the model performance as well as the loss_curve over the training history is saved in the folder 'out'.


## Results and summary

Different specifications have been implemented in the `build_model()` function to fine-tune the CNN architecture based on the VGG16 model, aiming to enhance performance in document classification tasks. 

These include:
- **Feature Extraction Enhancement**: Added a fully connected layer with 256 units and ReLU activation function to improve feature extraction capabilities.
- **Stability Improvement**: Applied batch normalization to stabilize training and improve convergence speed by normalizing the input to each layer.
- **Prevent Overfitting**: Implemented dropout regularization with a dropout rate of 0.5 to prevent overfitting by randomly dropping neurons during training.
- **Fine-Tuning Strategy**: Unfroze the last convolutional block of the pre-trained VGG16 model by allowing the last 4 layers to be trainable, enabling further adaptation to the specific document classification task.
- **Optimization Strategy**: Compiled the model with Stochastic Gradient Descent (SGD) optimizer, employing a lower learning rate of 0.0001 and momentum of 0.9 for fine-tuning, aiming for controlled updates and stable convergence during training.

These modifications are aimed at enhancing the model's ability to extract discriminative features from document images and improve classification accuracy. Gradually implementing these did indeed improve model performance, resulting in the final classification report and learning curve which can be found in the folder 'out'.

The classification report reveals high F1-scores across multiple classes, particularly notable in classes such as "Email" and "ADVE," where predictions were accurate approximately 94% and 89% of the time, respectively.
Instances of lower accuracy scores are evident in the "Scientific" class. One possible explanation for this could be the diverse nature of scientific disciplines, each adhering to distinct conventions, resulting in varying visual appearances of articles and papers. This diversity may pose challenges for the CNN in accurately classifying documents belonging to this category. Overall, the model demonstrates satisfactory performance in classifying documents across the 10 categories solely based on their visual appearance, as indicated by the weighted average f1-score of 0.79.

When visually inspecting the loss curves, they indicate a model which generally has a good fit to the data. 
Both train_loss and val_loss are decreasing gradually through the epochs, indicating that they are learning well from the data. Though there is a slight gap between the two curves, the distance hereof is rather consistent, indicating that the model is not just overfitting to the training data. Additionally, the curves seem to be flattening out at the end, indicating that the chosen number of epochs is good fit for this model and provided data.

