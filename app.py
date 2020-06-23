import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from joblib import load
import streamlit as st
from PIL import Image

ps = PorterStemmer()
pipeline = load('text_classification.joblib')
fake = Image.open('FAKE.png')
real = Image.open('REAL.png')

def create_corpus(text):
    review = re.sub('[^a-zA-Z]', ' ',text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    corpus = ' '.join(review)
    return corpus

def clean_text(text):
    replaced = re.sub("</?p[^>]*>", "", text)
    replaced = re.sub("\n","",replaced)
    replaced = re.sub("â","",replaced)
    final = re.sub('\W+',' ', replaced)
    return final

def main():
    
    st.title("Fake News Detection")
    st.markdown("Model is trained on this [data](https://www.kaggle.com/c/fake-news/data?select=train.csv)")
    html_temp = """
    <div style="background-color:#A816D6;padding:1.5px">
    <h5 style="color:black;text-align:center;">By Abhishek Mamdapure</h5>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    

    # st.success('The title of the article is :\n{}'.format(title))
    # st.success('The text of the article is :\n{}'.format(text))
    title = st.text_input("Title of the News"," ")
    author = st.text_input("Author of the News"," ")
    text = st.text_area("Text of the Article"," ")
    ip1 = clean_text(title + ' ' + author + ' ' + text)
    ip2 = create_corpus(ip1)
    
    if st.button("Predict"):        
        result = pipeline.predict([ip2])
        if result == 0:
            st.image(real,width=250,caption='RELIABLE NEWS')
        else:
            st.image(fake,width=250,caption='FAKE NEWS')
    
        
if __name__=='__main__':
    main()
