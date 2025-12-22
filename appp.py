import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ChatBot")
st.title("ChatBot")

# Load college data
with open("college_data.txt", "r") as f:
    corpus = f.read().lower().split("\n")

def chatbot_response(user_input):
    data = corpus + [user_input.lower()]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(data)

    similarity = cosine_similarity(tfidf[-1], tfidf)
    index = similarity.argsort()[0][-2]

    return corpus[index].capitalize()

user_input = st.text_input("Enter text to search:")

if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please enter a question")
    else:
        st.success(chatbot_response(user_input))
