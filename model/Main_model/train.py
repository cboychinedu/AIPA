#!/usr/bin/env python3

# Importing the necessary packages
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json

#
stemmer = LancasterStemmer()


# Extracting the data from the json dataset and
# Loading it into memory.
with open('model_1.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# Looping through the loaded json data and extract the tags
# needed
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Stemming the words and reducing it for our model
words = [stemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))

labels = sorted(labels)

# Creating an empty list to hold the training and output values
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# converting the values into numpy arrays and assinging it a variable.
training = np.array(training)
output = np.array(output)

# Developing the model and Training it
tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)

# Training and saving the Model
model.fit(training, output, n_epoch=20000, batch_size=5, show_metric=True)
model.save('model.tflearn')
