import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
tf=pickle.load(open('vectorizer.pkl','rb')) 

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    txts = " ".join(y)
    vector_input=tf.transform([txts]).toarray()
    return vector_input

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('voting.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    # vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(transformed_sms)
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")