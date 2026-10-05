# Project 2: Data Classification Using AI

A beginner-friendly machine learning project that uses the Iris dataset to build and evaluate a K-Nearest Neighbors (KNN) classifier for species prediction.

## Overview

This project demonstrates a standard supervised learning workflow:

- Load and inspect the dataset
- Split data into training and testing sets
- Standardize features using `StandardScaler`
- Tune the value of K using cross-validation
- Train a KNN classifier
- Evaluate performance with accuracy, F1-score, and confusion matrix
- Predict the class of a new flower sample

## Dataset

The project uses the built-in Iris dataset from `scikit-learn`.

### Features
- Sepal length
- Sepal width
- Petal length
- Petal width

### Classes
- Setosa
- Versicolour
- Virginica

## Files in the Repository

- `Basic_classification_model.py` — main project script that loads the dataset, trains the model, evaluates performance, and makes predictions
- `README.md` — project documentation

## Workflow

1. Load the Iris dataset
2. Inspect dataset shape and class distribution
3. Check missing values and summary statistics
4. Split the dataset into train and test sets
5. Scale the features using `StandardScaler`
6. Use cross-validation to select the best K value
7. Train the final KNN model
8. Measure performance using:
   - accuracy
   - macro F1-score
   - confusion matrix
   - classification report
9. Save plots for model selection and confusion matrix
10. Predict the species of a new sample

## Requirements

Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run the Project

Execute the script from the project folder using the actual file name in the repository:

```bash
python Basic_classification_model.py
```

This will print evaluation metrics and generate image files:

- `k_selection.png`
- `confusion_matrix.png`

## Example Output

The script provides:

- dataset shape and summary
- class balance information
- training/testing sample counts
- best K value selected through validation
- accuracy and F1-score
- confusion matrix
- classification report
- prediction for a new flower instance

## Notes

- The data is scaled only after splitting to avoid data leakage.
- K is chosen using cross-validation on the training data.
- This project is best suited for learning the basics of classification and evaluation in machine learning.

## Author

RITESH KUMAR SRIVASTAVA
