
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model
model_path = 'trained_model.sav'  # Ensure that this path is correct
loaded_model = pickle.load(open(model_path, 'rb'))

# Load the fitted TfidfVectorizer
vectorizer_path = 'vectorizer.pkl'  # Path to the saved TfidfVectorizer
loaded_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Streamlit App
st.title('Twitter Sentiment Analysis')

# Input field for entering tweet
tweet_input = st.text_area("Enter the tweet for sentiment analysis:")

# Predict button
if st.button('Predict Sentiment'):

    # Preprocess and vectorize the input
    if tweet_input:
        tweet_vector = loaded_vectorizer.transform([tweet_input]).reshape(1, -1)
        
        # Predict sentiment
        prediction = loaded_model.predict(tweet_vector)
        
        # Display the result
        if prediction[0] == 0:
            st.write("The sentiment is **Negative**.")
        else:
            st.write("The sentiment is **Positive**.")
    else:
        st.write("Please enter a tweet for analysis.")
