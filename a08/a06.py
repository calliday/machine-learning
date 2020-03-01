#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:56:29 2020

@author: Caleb
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from CustomNN import normalizeData
from CustomNN import BasicNN
from matplotlib import pyplot


'''
# Get the iris data
iris = datasets.load_iris()
targets = np.array(iris.target)
data = np.array(iris.data)
'''

# Get the banknote data
auth = np.loadtxt("data_banknote_authentication.txt", delimiter=',')
data = auth[400:1000,:-1]
targets = auth[400:1000,-1:]

'''
# Get the letter recognition data
letters = np.loadtxt("letters.csv", delimiter=',')
data = letters[:,1:]
targets = letters[:,:1]
'''

# Preprocessing the data
normalizeData(data)

# Get the train/test split
data_train, data_test, target_train, target_test = train_test_split(data, targets, train_size=0.3)

''' Graph the training '''
# Setup the learning network

nn = BasicNN()
nn.fit(data_train, target_train, [], -1)
predictions = None
learning_data = []

# Keep track of progress (so I know that it's working in the background)
count = 0
epochs = 100
print("Progress: 0%")
for i in range(0, epochs):
    if i / epochs > 0.25 and count == 0:
        print("Progress: 25%")
        count += 1
    elif i / epochs > 0.5 and count == 1:
        print("Progress: 50%")
        count += 1
    elif i / epochs > 0.75 and count == 2:
        print("Progress: 75%")
        count += 1
    nn.train(0.2, 1)
    predictions = nn.predict(data_train)[np.newaxis].T
    learning_data.append(np.mean(predictions == target_train) * 100)

# Plot and show the learning data
pyplot.plot(learning_data)
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy %")
pyplot.show()

print

# Predict the test values and show accuracy
predictions = nn.predict(data_test)[np.newaxis].T
print("Final Accuracy: ", np.mean(predictions == target_test) * 100)

'''
# Batch Train the network
bnn = BasicNN()
bnn.fit(data_train, target_train, [8], -1)

# Keep track of learning iterations
learning_data = []
for _ in range(0, 100):
    bnn.batchTrain(0.2, 1, 5)
    predictions = bnn.predict(data_train)[np.newaxis].T
    learning_data.append(np.mean(predictions == target_train) * 100)    

predictions = bnn.predict(data_test)
tpredictions = predictions[np.newaxis].T
combined = np.append(tpredictions, target_test[np.newaxis].T, 1)
print(combined)

# Plot and show the learning data
pyplot.plot(learning_data)
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy %")
pyplot.show()

print("Accuracy: = ", np.mean(predictions == target_test) * 100)
'''









