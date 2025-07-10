# 📧 Full Streamlit NLP Spam Detector App (from scratch, ready to deploy)

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')

# ✅ 1. Load dataset
spam_df = pd.read_csv('emails.csv')

# ✅ 2. Text cleaning function
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
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(spam_df['clean_text'])
y = spam_df['spam']

# ✅ 4. Train model
model = MultinomialNB()
model.fit(X, y)

# ✅ 5. Streamlit app UI
st.title("📧 Email Spam Detector")
st.write("Enter any email text below, click **Predict**, and see if it's spam or not.")

user_input = st.text_area("✏️ Type your email here:")

if st.button("🚀 Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
    st.subheader("Result:")
    st.write(result)

st.markdown("---")
st.caption("Built with Streamlit and scikit-learn.")
