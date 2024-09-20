import pickle
import warnings
import streamlit as st
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


# Define customised_pre_processor function
def customised_pre_processor(review):
    review = review.strip()

    # Convert to lowercase
    review = review.lower()

    # Remove non-English characters
    review = re.sub(r'[^a-zA-Z\s]', '', review)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    review = ' '.join([word for word in word_tokenize(review) if word.lower() not in stop_words])

    # Apply stemming
    stemmer = PorterStemmer()
    review = ' '.join([stemmer.stem(word) for word in word_tokenize(review)])
    return review


with open("../resource/output/vetorizer.pkl", "rb") as pickle_in:
    tfidf_vectorizer = pickle.load(pickle_in)

with open("../resource/output/tfidvetorizer.pkl", "rb") as pickle_in:
    tfid = pickle.load(pickle_in)

with open("../resource/output/vectorizer.pkl", "rb") as pickle_in:
    vectorizer = pickle.load(pickle_in)
  
# Load models and vectorizer
with open("../resource/output/linear.pkl", "rb") as pickle_in:
    clf = pickle.load(pickle_in)

with open("../resource/output/logistic.pkl", "rb") as pickle_in:
    model = pickle.load(pickle_in)

with open("../resource/output/bayesian.pkl", "rb") as pickle_in:
    bayesian = pickle.load(pickle_in)

# Define sentiment prediction function
def linear_analysis(text):
    start = time.time()
    # Preprocess the input text
    preprocessed_text = customised_pre_processor(text)
    
    # Transform the text using the loaded tfidf_vectorizer
    # features = tfidf_vectorizer.transform([preprocessed_text])
    features = tfidf_vectorizer.transform([preprocessed_text])

    # Predict sentiment using the loaded classifier
    prediction = clf.predict(features)[0]
    end = time.time()
    process_time = end - start
    
    # Return the sentiment label and processing time
    return "Negative" if prediction == 1 else "Positive", process_time

def logisticRegression_analysis(text):
    start = time.time()
    # Preprocess the input text
    preprocessed_text = customised_pre_processor(text)
    
    # Transform the text using the loaded tfidf_vectorizer
    features = tfid.transform([preprocessed_text])
    
    # Predict sentiment using the loaded classifier
    prediction = model.predict(features)[0]
    end = time.time()
    process_time = end - start
    
    # Return the sentiment label and processing time
    return "Negative" if prediction == 1 else "Positive", process_time

def bayesian_analysis(text):
    start = time.time()
    preprocessed_text = customised_pre_processor(text)

    features = vectorizer.transform([preprocessed_text])

    prediction = bayesian.predict(features)[0]
    end = time.time()
    process_time = end - start
    
    return "Positive" if prediction == "positive" else "Negative", process_time

# Define Streamlit app
def Input_Output():
    st.title("Sentiment Analysis")

    st.markdown("You are using Streamlit...", unsafe_allow_html=True)
    user_input = st.text_input("Enter your review", "")
    
    result = ""
    if st.button("Click here to Predict"):
        result, process_time = linear_analysis(user_input)
        result1, process_time1 = logisticRegression_analysis(user_input)
        result2, process_time2 = bayesian_analysis(user_input)
    
        st.balloons()
        st.success(f"Linear Model: {result}")
        st.success(f"Execution time = {process_time * 1000:.2f} milliseconds")
        st.success(f"Logistic Regression Model: {result1}")
        st.success(f"Execution time = {process_time1 * 1000:.2f} milliseconds")
        st.success(f"Bayesian Model: {result2}")
        st.success(f"Execution time = {process_time2 * 1000:.2f} milliseconds")

    # Create a bar plot for execution times
        fig, ax = plt.subplots()  # Create a new figure and axes
        ax.bar(['Result 1', 'Result 2', 'Result 3'],
            [process_time * 1000, process_time1 * 1000, process_time2 * 1000])
        ax.set_xlabel('Model')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Comparison of Execution Times')
        st.pyplot(fig)  # Pass the figure to st.pyplot()

if __name__ == '__main__':
    Input_Output()
