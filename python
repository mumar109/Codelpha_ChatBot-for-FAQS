# FAQ Chatbot using NLP (TF-IDF + Cosine Similarity)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# Step 1: Create FAQ Dataset
# -----------------------------
faq_data = {
    "question": [
        "What is your return policy?",
        "How can I track my order?",
        "Do you offer international shipping?",
        "How do I contact customer support?",
        "What payment methods are accepted?"
    ],
    "answer": [
        "You can return items within 30 days of purchase.",
        "You can track your order using the tracking link sent to your email.",
        "Yes, we ship to most countries worldwide.",
        "You can contact support via email or our help center.",
        "We accept credit cards, debit cards, and PayPal."
    ]
}

df = pd.DataFrame(faq_data)

# -----------------------------
# Step 2: Preprocessing Function
# -----------------------------
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [word for word in tokens if word not in string.punctuation]
    return " ".join(tokens)

# Apply preprocessing
df['processed'] = df['question'].apply(preprocess)

# -----------------------------
# Step 3: Vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed'])

# -----------------------------
# Step 4: Chatbot Function
# -----------------------------
def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])

    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)
    score = similarity[0][index]

    if score < 0.3:
        return "Sorry, I couldn't understand your question."

    return df['answer'][index]

# -----------------------------
# Step 5: Chat Loop
# -----------------------------
print("FAQ Chatbot (type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)
    print("Bot:", response)


# Save this as app.py and run: streamlit run app.py

"""
import streamlit as st

st.title("FAQ Chatbot")

user_input = st.text_input("Ask a question:")

if user_input:
    response = get_response(user_input)
    st.write("Answer:", response)
"""
