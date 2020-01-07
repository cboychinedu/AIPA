#!/usr/bin/env python3 

# Author: Mbonu Chinedu Endurance 
# Company: Analytics Ingtelligence 
# Description: 
# Email: 
# Date Created: 

# Importing the necessary Packages 
import os 
import nltk 
import tflearn 
import datetime 
import json 
import string 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import tensorflow as tf 
import wikipedia as wi 
from nltk.corpus import stopwords 
from nltk.stem.lancaster import LancasterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from spellchecker import SpellChecker 
from time import sleep 




# loading the json Dataset into memory 
# Specifying the path to the dataset folder 
dataset = 'model/Main_model/dataset.json'
# Then loading the dataset into memory using the open method. 
with open(dataset) as file:
    data = json.load(file)

# initializing the stemmer module  
stemmer = LancasterStemmer()
# Assigning a variable called spell to check and correct the spelling of the input word
# or sentences passed into the system.
spell = SpellChecker()

# setting a try and exception method to load the pickle file into memory 
# and create a new one. 
try:
    with open('dataset_folder/data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Converting the values in the json dataset into words and then tokenize them and append 
    # them into the created lists above 
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
        # setting an if/else statement to check if the words is present in the labels 
        # and skip it 
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    # convert the words into lower case letters and then stem them and save them into a variable 
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    # Assigning the varibale labels to hold the sorted words in the list labels 
    labels = sorted(labels)
    # Creating an empty list to hold both the training and output list 
    training = []
    output = []
    # Assigning the length of words in the labels variables into a new variable called output 
    out_empty = [0 for _ in range(len(labels))]
    # Creating a for loop to loop through the values in docs_x and enumerate them and create a new empty list called bag
    # that would hold the numerical values for the words.
    # Then Stemming the words in docs_x and turning them into lower case letters before appending the index value
    # to the bag of words list
    for x, doc in enumerate(docs_x):
        bag = []
        # converting the word into lower case and stemming them 
        # saving the stemmed words into a list called wrds 
        wrds = [stemmer.stem(w.lower()) for w in doc]
        # Creating loop that world loop through the word in words and append 1 if 
        # present and zero if the word is not present 
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1 
        # Then appending the values or numerical values in the bag of words list into a training list that 
        # would be used to train the machine learning classifier 
        training.append(bag)
        output.append(output_row)
    # converting the values or numbers of both the training and output list 
    # into a numpy matrix and saving it into a list called training 
    training = np.array(training)
    output = np.array(output)
    # saving the cleaned dataset into a pickle file 
    with open('dataset_folder/data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)
# Building the model by adding 8 neurons for 3 layers and resetting the tf graph 
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net) 
# Creating the model with tflearn and passing the neurons through DNN blob 
model = tflearn.DNN(net)
# loading the model into memory and saving it as model for predictions 
try:
    model.load('model/Main_model/model.tflearn')
except:
    # Training the model with 1000 epochs and batch size of 8 to generate a model file 
    model.fit(training, output, n_epoch=10000, batch_size=5, show_metric=True)
    model.save('model/Main_model/model.tflearn')
# Creating a function for taking in words and stemming them into simpler words, then 
# saving its representative index number into a variable called a bag of words 
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    # converting the sentences into words and saving them into a variable called s_words 
    # then stemming the words to get a single and simplified word. 
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    # Creating a loop to loop through the words and enumerate them, then 
    # append 1 to the bag of words list if the word is present and 0 if not present 
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1 
        # Then return the list as a numpy array of matrix 
        return np.array(bag)

# Creating a function to check if the image present in the directory is 
# more than 5 , if more than 5 delete all the images and save the new image 
def image_check():
    # getting the path of the current working directory 
    path = os.pathcwd() 
    # getting the names and listing the numbers of the files present in the directory 
    file_count = os.listdir(path)
    # getting the number of the files 
    file_count = len(file_count)
    # Returning the value 
    return file_count 


# Creating a fucntion to start the Personal Assistant. 
def chat(message):
    # Converting the sentences in the message variable into lower case letters. 
    message = message.lower()
    # Creating an empty list to hold the corrected words 
    _corrected_word = []
    # Correcting the words in message variable before sending it into the model for prediction 
    for wrong_words in message.split():
        _corrected = spell.correction(wrong_words)
        _corrected_word.append(_corrected)
    # Then joining the corrected words in the list to a simple sentence 
    corrected_word = ' '.join(_corrected_word)
    input_message = corrected_word 
    # passing the corrected sentences into the Neural net to make predictions on the actual tags its was 
    # trained on. 
    results = model.predict([bag_of_words(input_message, words)])
    # finding the index value of the predicted results 
    results_index = np.argmax(results)
    # to then find the respective predicted tag, we place the index inside of the list 
    # to output the predicted tag label. 
    tag = labels[results_index] 

    # Performing some certain functions if the predicted tag is a qustion. 
    if tag == 'question':
        # Creating a list to hold the kadris question tags respectively. 
        kadris_tags = [
        'reg_is_insured', 'reg_not_insured', 'reg_not_license', 'reg_is_license',
        'vehicle_not_license', 'reg_not_mot', 'reg_is_mot', 'vehicle_is_mot', 'vehicle_is_insured',
        'vehicle_is_insured', 'vehicle_not_insured', 'vehicle_is_license', 'vehicle_not_mot', 'vehicle_not_mot', 
        'invalid_input', 'invalid_input', 'vehicle_is_license ']
        # Importing the model for the question classifier 
        from model.question.question import question_classifier 
        # passing the question message into the question message classifier 
        msg = question_classifier(input_message)
        # Checking to see if the predicted message from the question classifer is present in the 
        # kadris tag list. 
        if msg in kadris_tags:
            pass

    # 
    elif tag == 'conversation':
        # Importing the model for the conversation classifier 
        from model.conversation.conversation import conversation_classifier
        # passing the conversational message into the conversational message classifier 
        msg = conversation_classifier(input_message)
        print(msg)
        pass 

    # 
    else:
        pass 