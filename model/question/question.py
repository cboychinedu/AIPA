#!/usr/bin/env python3 

# Description: This is an intelligent bot that uses Naive Bayes classifier for predictions 
# Author: Mbonu Chinedu Endurance 
# Company: Analytics Intelligence 
# Email: chinedu.mbonu@analyticsintelligence.com 
# Date Created: 15-sept-2019 

# Importing the necessary packages 
import string 
import joblib 
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.corpus import stopwords 
from nltk import PorterStemmer as Stemmer 
from sklearn.linear_model import LogisticRegression 

# Defining a function to clean the text and return it as a string 
def process(text):
    # turn the text into lowercase 
    text = text.lower() 
    # removing the punctuation 
    text = ''.join([t for t in text if t not in string.punctuation])
    # removing the stopwords 
    text = [t for t in  text.split() if t not in stopwords.words('english')]
    # stemming the words 
    stemmer = Stemmer() 
    text = [stemmer.stem(t) for t in text]
    # returning the token list 
    return text 

# passing the stemmed words into a vectorizer to convert the words 
# into vectors and hold numbers of the word count of each individual words 
vectorizer = CountVectorizer()

# Building the model using Naive bayes classifier 
question_model = LogisticRegression()

# loading the saved model from disk 
filename = 'model/question/question_model.sav'

# load the model from disk 
question_model, vectorizer = joblib.load(filename)

# Defining a function to run and use the model for classification and return the 
# values back to the user 
def question_classifier(s):
    question = [s]
    question = vectorizer.transform(question)
    result = question_model.predict(question)[0]
    return result 

