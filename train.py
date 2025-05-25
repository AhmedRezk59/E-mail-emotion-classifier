import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from preprocess import preprocessing_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV,cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from persistence_manager import PersistenceManager

### Load the dataset
persistence_manager = PersistenceManager()
df = persistence_manager.load()
df.dropna(subset="emotion", inplace=True)

X = df[["text"]]
y = df["emotion"].values

# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and feature extraction
svc_pipe = Pipeline([
    ("preprocess" , preprocessing_pipeline),
    ("svc" , SVC(kernel='linear', probability=True))
])

### Perform cross-validation to evaluate the model
scores = cross_val_score(svc_pipe, x_train, y_train, cv=5, scoring='accuracy')
print(f"F1 Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


### Perform hyperparameter tuning using RandomizedSearchCV
param_dist = {
    "svc__C": [0.1, 1, 10],
    "svc__gamma": [0.001, 0.01, 0.1],
    "svc__kernel": ["linear", "rbf"]
}

cv = RandomizedSearchCV(
    svc_pipe,
    param_dist,
    n_iter=30,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

cv.fit(x_train, y_train)

### Print the best parameters and score
print("Best parameters found: ", cv.best_params_)
print("Best cross-validation score: ", cv.best_score_)

#### Evaluate the model on the test set
y_pred = cv.predict(x_test)
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

### Save the best model
best_model = cv.best_estimator_
persistence_manager.save_model(best_model)