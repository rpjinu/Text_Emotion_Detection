import streamlit as st
import joblib
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Ensure that stopwords are downloaded
nltk.download('stopwords')

# Initialize the PorterStemmer
port_stem = PorterStemmer()

# Function to preprocess text
def stemming(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Split into words
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]  # Remove stopwords
    text = ' '.join(text)  # Rejoin words into a string
    return text

# Define the TextPreprocessor class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_func):
        self.preprocessing_func = preprocessing_func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, list):
            return [self.preprocessing_func(text) for text in X]
        return X.apply(self.preprocessing_func)

# Load the trained model pipeline
MODEL_PATH = r'D:\Dawnloads\emotion_text_dataset\emotion_classification_model.pkl'

@st.cache_resource
def load_model():
    """Load the pre-trained model pipeline."""
    return joblib.load(MODEL_PATH)

model_pipeline = load_model()

# Emotion mapping
emotion_mapping = {
    0: 'joy',
    1: 'sadness',
    2: 'anger',
    3: 'fear',
    4: 'love',
    5: 'surprise'
}

# Streamlit app UI
st.title("Emotion Prediction website")
st.write("Enter a sentence to predict its emotion.")

# Input text box
user_input = st.text_input("Enter a sentence:", "")

if st.button("Predict"):
    if user_input:
        try:
            # Predict emotion
            prediction_numeric = model_pipeline.predict([user_input.strip()])[0]
            prediction_label = emotion_mapping.get(prediction_numeric, "unknown")
            st.write(f"Predicted Emotion: **{prediction_label}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter a sentence.")



