#!/usr/bin/env python3 

# Author: Mbonu Chinedum Endurance 
# Company: Analytics Intelligence 
# Email: chinedu.mbonu@analyticsintelligence.com 
# Date Created: 15-sept-2019 

# Importing the necessary packages 
import joblib 
import string 
from nltk.corpus import stopwords 
from nltk import PorterStemmer as Stemmer 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer 

# Creating a function for cleaning the text 
def process(text):
    # turn the texts into lowercase 
    text = text.lower() 
    # remobe the punctuation 
    text = ''.join([t for t in text if t not in string.punctuation])
    # remove the stopwords 
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # stemming the words 
    stemmer = Stemmer() 
    text = [stemmer.stem(t) for t in text]
    # return the token list 
    return text 

# specifying the vectorizer to use 
vectorizer = CountVectorizer()

# Building the model using Naive Bayes classifier 
conversation_model = LogisticRegression()

# loading the saved model from disk into memory 
filename = 'model/conversation/conversational_model.sav'
conversation_model, vectorizer = joblib.load(filename)

# defining a function to run and use the model for classification 
def conversation_classifier(s):
    conversation = [s]
    conversation = vectorizer.transform(conversation)
    result = conversation_model.predict(conversation)[0]
    return result 


