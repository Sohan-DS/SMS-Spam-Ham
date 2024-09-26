import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import requests

# Download the model file from GitHub
@st.cache_data
def download_model():
    url = "https://github.com/Sohan-DS/SMS-Spam-Ham/raw/main/spam_mod.h5"
    response = requests.get(url)
    with open("spam_mod.h5", "wb") as f:
        f.write(response.content)

# Call the function to download the model
download_model()

# Load the saved model
model = tf.keras.models.load_model("spam_mod.h5")

# Define a function to initialize the tokenizer
@st.cache_data
def initialize_tokenizer():
    tokenizer = Tokenizer()
    return tokenizer

# Initialize the tokenizer
tokenizer = initialize_tokenizer()

# Function to preprocess input text
def preprocess_text(input_text, tokenizer, max_length=10):
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequence

# Streamlit app UI
st.title("SMS Spam Classifier")

st.write("""
### Enter an SMS message to classify it as spam or ham:
""")

# Text input box
user_input = st.text_input("Enter SMS text here:")

# Button for prediction
if st.button("Classify"):
    if user_input.strip() == "":
        st.write("Please enter a valid SMS message.")
    else:
        # Preprocess the input text
        preprocessed_text = preprocess_text(user_input, tokenizer)
        
        # Make prediction
        prediction = model.predict(preprocessed_text)
        
        # Display result
        if prediction > 0.5:
            st.write("🔴 This message is likely **Spam**.")
        else:
            st.write("🟢 This message is likely **Ham** (Not Spam).")
