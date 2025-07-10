# 📧 Streamlit Spam Detector App

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')

# ✅ 1. Load dataset
spam_df = pd.read_csv('emails.csv')

# ✅ 2. Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

spam_df['clean_text'] = spam_df['text'].apply(clean_text)

# ✅ 3. Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(spam_df['clean_text'])
y = spam_df['spam']

# ✅ 4. Train model (simple Naive Bayes as in your code)
model = MultinomialNB()
model.fit(X, y)

# ✅ 5. Streamlit app interface
st.title("📧 Email Spam Detector (Naive Bayes)")
st.write("Enter email text below and click Predict:")

user_input = st.text_area("✏️ Your email:")

if st.button("🚀 Predict"):
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]
    st.subheader("Result:")
    st.write("🚫 Spam" if prediction == 1 else "✅ Not Spam")

st.markdown("---")
st.caption("Built from scratch based on original NLP spam detection code.")
