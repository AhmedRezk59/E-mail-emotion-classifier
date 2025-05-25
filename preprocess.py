import re
import neattext.functions as ntf
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

lemmatizer = WordNetLemmatizer()

# Ensure that the necessary NLTK resources are downloaded
def ensure_nltk_resource(resource_path, download_name=None):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        download_target = download_name if download_name else resource_path.split('/')[-1]
        nltk.download(download_target, quiet=True)

ensure_nltk_resource('corpora/stopwords')
ensure_nltk_resource('corpora/wordnet')

stop_words = set(stopwords.words('english')) - {"not" , "no", "never", "none", "nor", "nobody", "nothing", "nowhere", "noone"}

def clean_text(text) -> str:
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing digits
    3. Removing punctuation
    4. Removing special characters
    5. Removing stop words

    Args:
        text: text to be cleaned

    Returns:
        str: text after cleaning
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+","",text)
    text = ntf.remove_puncts(text)
    text = ntf.remove_special_characters(text)
    
    words = [word for word in text.split() if word not in stop_words]
    
    return " ".join(lemmatizer.lemmatize(word) for word in words)

def apply_clean_text(df:pd.DataFrame) -> pd.DataFrame:
    """
    Applies the clean_text function to the 'text' column of the DataFrame.

    Args:
        df: DataFrame containing a 'text' column

    Returns:
        DataFrame: DataFrame with cleaned text
    """
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    df = df.copy()
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df['cleaned_text']


def fillna_func(x):
   return x.assign(text = x["text"].fillna(""))

def clean_text_func(x):
    return apply_clean_text(x)

fillna = FunctionTransformer(fillna_func, validate=False)
clean_text_transformer = FunctionTransformer(clean_text_func, validate=False)
vectorizer = TfidfVectorizer(max_features=5000)

preprocessing_pipeline = Pipeline([
    ("fillna" , fillna),
    ("clean_text" , clean_text_transformer),
    ("vectorizer" , vectorizer)
])