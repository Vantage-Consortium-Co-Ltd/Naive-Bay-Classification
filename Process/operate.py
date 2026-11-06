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

def naive_bayes():
    # Load the dataset
    data = load_data()
    
    # Preprocess the dataset
    processed_data = preprocess_dataset(data)
    
    # Split the dataset into training and testing sets
    train_data, test_data = split_data(processed_data)
    
    # Train the Naive Bayes model
    model = train_model(train_data)
    
    # Evaluate the model on the test set
    accuracy = evaluate_model(model, test_data)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


