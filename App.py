# Importing required libraries
import nltk  # Natural Language Toolkit for text processing
nltk.download('punkt')  # Downloading tokenizer model
nltk.download('punkt_tab')  # Downloading additional tokenization resources
nltk.download('stopwords')  # Downloading list of stopwords

import streamlit as st  # For building the web app interface
import pickle  # For loading pre-trained models and vectorizers
import string  # For handling punctuation
from nltk.corpus import stopwords  # To filter out common stopwords
from nltk.stem.porter import PorterStemmer  # For stemming words
from sklearn.feature_extraction.text import TfidfVectorizer  # For vectorizing text

# Initializing the PorterStemmer for stemming
ps = PorterStemmer()

# Function to preprocess and transform input text
def transform_text(text):
    # Step 1: Convert all text to lowercase
    text = text.lower()
    
    # Step 2: Tokenize the text into words
    text = nltk.word_tokenize(text)

    # Step 3: Remove non-alphanumeric tokens
    y = []
    for i in text:
        if i.isalnum():  # Check if the token is alphanumeric
            y.append(i)

    text = y[:]  # Replace the original list with filtered tokens
    y.clear()  # Clear the temporary list

    # Step 4: Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]  # Replace the original list with filtered tokens
    y.clear()  # Clear the temporary list

    # Step 5: Stem the words
    for i in text:
        y.append(ps.stem(i))  # Apply stemming to each token

    # Join the processed words into a single string and return
    return " ".join(y)

# Load the pre-trained vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))  # Load TF-IDF vectorizer
model = pickle.load(open("model.pkl", 'rb'))  # Load classification model

# Streamlit web app setup
st.title("SMS Spam Detection Model")  # Title of the app
st.write("*Made by Chayan Jana*")  # App subtitle/author credit

# Input field for SMS
input_sms = st.text_input("Enter the SMS")  # Prompt user to enter SMS text

# Button to trigger prediction
if st.button('Predict'):
    # Step 1: Preprocess the input SMS
    transformed_sms = transform_text(input_sms)
    
    # Step 2: Vectorize the transformed SMS text
    vector_input = tk.transform([transformed_sms])
    
    # Step 3: Predict whether the SMS is spam or not
    result = model.predict(vector_input)[0]
    
    # Step 4: Display the result
    if result == 1:
        st.header("Spam")  # Display "Spam" if predicted as spam
    else:
        st.header("Not Spam")  # Display "Not Spam" if predicted as not spam
