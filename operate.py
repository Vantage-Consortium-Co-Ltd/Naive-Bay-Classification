import numpy as np
from pythainlp import word_tokenize

def preprocess_text(text):
    # Tokenize the text using PyThaiNLP
    tokens = word_tokenize(text, engine="longest")
    
    # Remove punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]
    
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    
    return tokens

def preprocess_dataset(dataset):
    # Apply preprocessing to the 'review' column of the dataset
    dataset['processed_review'] = dataset['review'].apply(preprocess_text)
    
    return dataset

