import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = """
Hello I am an AI chatbot
Artificial intelligence is the simulation of human intelligence
Machine learning is a subset of AI
Natural language processing allows computers to understand text
Python is widely used for AI
Chatbots simulate human conversation
"""

sent_tokens = nltk.sent_tokenize(corpus)
lemmer = nltk.stem.WordNetLemmatizer()

def LemNormalize(text):
    return [lemmer.lemmatize(word.lower())
            for word in nltk.word_tokenize(text)
            if word not in string.punctuation]

def get_response(user_input):
    sent_tokens.append(user_input)
    tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    matrix = tfidf.fit_transform(sent_tokens)
    similarity = cosine_similarity(matrix[-1], matrix)
    idx = similarity.argsort()[0][-2]
    score = similarity.flatten()[-2]
    sent_tokens.pop()

    if score == 0:
        return "Sorry, enaku puriyala 😕"
    else:
        return sent_tokens[idx]
