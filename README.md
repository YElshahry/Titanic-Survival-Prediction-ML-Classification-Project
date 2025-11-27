# Titanic Survival Prediction — Machine Learning Classification Project

This repository contains an end-to-end machine learning project built on the Kaggle Titanic dataset.

The goal is to predict passenger survival using data preprocessing, feature engineering, model training, and hyperparameter tuning.

This project uses:
- Exploratory Data Analysis (EDA)
- Scikit-learn Pipelines
- ColumnTransformer
- Logistic Regression, Random Forest, SVM, KNN
- GridSearchCV hyperparameter tuning
- Model evaluation (accuracy, classification report, confusion matrix)
- Saving the best model with joblib
- Modular Python scripts in `/src`


# Overview of the ML Pipeline


# 1- Data Preprocessing
Located in: `src/data_preprocessing.py`

- Handle missing values
- Select key features (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`)
- One-hot encode categorical variables
- Scale or passthrough numeric variables
- Outputs X (features) and y (target)

# 2- Model Training
Located in: `src/model_training.py`

- Trains multiple models:
  - Logistic Regression  
  - Random Forest  
  - Support Vector Classifier  
  - KNN  
- Cross-validation using `GridSearchCV`
- Tracks accuracy of each model
- Saves results

# 3- Model Evaluation
Located in: `src/utils.py`

- Prints:
  - Accuracy  
  - Classification Report  
  - Confusion Matrix  

# 4- Model Tuning
Inside: `03_tuning.ipynb`

- Grid search spaces for best hyperparameters
- Best model stored as:
    best_titanic_model.pkl


# How to Run the Project


# 1- Install dependencies
pip install -r requirements.txt

# 2- Run the notebooks
Start with:
    notebooks/01_eda.ipynb
    notebooks/02_modelling.ipynb

# 3- Train or tune models
    notebooks/03_tuning.ipynb

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
