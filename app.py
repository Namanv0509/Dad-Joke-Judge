import streamlit as st
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

st.set_page_config(page_title="Dad Joke Judge")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\NAMAN VERMA\Downloads\SOM_FINAL_PROJECT\reddit_dadjokes.csv")
    drop = ['author', 'url', 'date']
    df.drop(drop, axis=1, inplace=True)
    return df

df = load_data()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_and_lemmatize(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

df['preprocessed_jokes'] = df['joke'].apply(preprocess_and_lemmatize)

all_jokes = ' '.join([' '.join(joke) for joke in df['preprocessed_jokes']])

word_counts = Counter(all_jokes.split())

frequency_threshold = 5

frequent_words = [word for word, count in word_counts.items() if count >= frequency_threshold]

def predict_dad_joke(joke):
    preprocessed_joke = preprocess_and_lemmatize(joke)

    high_frequency_count = 0
    low_frequency_count = 0

    for word in preprocessed_joke:
        if word in frequent_words:
            high_frequency_count += 1
        else:
            low_frequency_count += 1

    count_words = len(preprocessed_joke)

    if high_frequency_count > 3 and high_frequency_count > 0.5 * count_words:
        return "Dad Joke"
    else:
        return "Not a Dad Joke"

st.title("Dad Joke Judge")
st.write("Welcome to the Dad Joke Judge!")

test_joke = st.text_area("Enter a joke:")

if st.button("Judge"):
    result = predict_dad_joke(test_joke)
    st.write(f"The joke: '{test_joke}' is classified as: {result}")
