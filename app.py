# Importing Library
import nltk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import streamlit as st
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def preprocess_text(text):
    # Converting to lower case
    text = text.lower()
    # Word tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    retList=[]
    for i in text:
        if i.isalnum():
            retList.append(i)

    # Removing Stop words and punctuation
    text = retList[:]
    retList.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            retList.append(i)

    # Stemming
    ps=PorterStemmer()
    text = retList[:]
    retList.clear()
    for i in text:
        retList.append(ps.stem(i))
    
    return " ".join(retList)

def load_dataset(): 
    global df
    # Loading dataset credits
    df=pd.read_csv("spam.csv",encoding = "ISO-8859-1")
    # drop last 3 columns
    df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
    # rename columns
    df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
    # label encoder
    le=LabelEncoder()
    df['target']=le.fit_transform(df['target'])
    # remove duplicate values
    df=df.drop_duplicates(keep='first')
    df['transformed_text']=df['text'].apply(preprocess_text)

def create_model():
    global cv_imp
    global voting
    cv_imp=TfidfVectorizer(max_features=3000)
    # X and y
    X=cv_imp.fit_transform(df['transformed_text']).toarray()
    y=df['target'].values
    # train_test_split
    X_train,X_test, y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=2)

    mnb=MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred=mnb.predict(X_test)
    print(f"Model Accuracy={accuracy_score(y_test, y_pred)} and Precision={precision_score(y_test, y_pred)}")  
    return cv_imp, mnb
    
@st.cache_resource
def make_model():    
    global tfidf
    global model
    load_dataset()    
    tfidf, model = create_model()
    return tfidf, model
    
tfidf, model = make_model()
st.title("SPAM Classifier")    
string_text = st.text_area('Enter your mail', height=150)
if st.button('Check'):
    transformed_text=preprocess_text(string_text)
    vector_input=tfidf.transform([transformed_text])
    y_pred_text = model.predict(vector_input)
    if y_pred_text == 1:
        st.write('SPAM ')
    else:
        st.write('NOT SPAM ')    
