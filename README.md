# Emotion Classification ML Pipeline

This project implements a machine learning pipeline for emotion classification from text data. It uses Python, scikit-learn, and NLTK for preprocessing, feature extraction, model training, and evaluation.

## Features

- **Data Loading & Persistence:**  
  Uses a `PersistenceManager` class to load and save datasets and trained models.

- **Preprocessing:**  
  - Cleans text by lowercasing, removing digits, punctuation, special characters, and stop words (with some negations kept).
  - Applies lemmatization.
  - Handles missing values in the text column.
  - Uses a scikit-learn `Pipeline` for modular preprocessing.

- **Feature Extraction:**  
  - Uses `TfidfVectorizer` to convert cleaned text into numerical features.

- **Model Training:**  
  - Trains a Support Vector Classifier (SVC) with a pipeline that includes preprocessing and feature extraction.
  - Performs cross-validation to estimate model performance.
  - Uses `RandomizedSearchCV` for hyperparameter tuning.

- **Evaluation:**  
  - Prints F1 score, accuracy, classification report, and confusion matrix on the test set.

- **Model Saving:**  
  - Saves the best trained model for later use.

## Usage

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Prepare your dataset:**  
   Ensure your data file contains at least two columns: `text` and `emotion`.

3. **Run training:**
    ```bash
    python train.py
    ```

4. **Model output:**  
   The best model is saved using the `PersistenceManager`.

5. **Streamlit:**
   ```bash
    streamlit run main.py
   ```
## File Structure

- `train.py` — Main script for training, validation, and model saving.
- `preprocess.py` — Preprocessing functions and pipeline components.
- `persistence_manager.py` — Handles loading and saving of data/models.
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- nltk
- matplotlib

## Notes

- The pipeline is designed to be robust to missing text values.
- All preprocessing steps are encapsulated in the pipeline for reproducibility.
- Hyperparameter search is performed for SVC’s `C`, `gamma`, and `kernel` parameters.

---