#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:13:13 2020

@author: Caleb

DESCRIPTION: This program includes a custom KNN machine learning class as well
as the compared resulting predictions of a KNN machine built into sklearn
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from customClassifier import CustomKNeighbors

# Load iris data and split into testing/training
iris = datasets.load_iris()
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, train_size=0.3)

# Deciding K as one variable

k = 21

""" 
CUSTOM KNN
"""

custom_classifier = CustomKNeighbors(k)
custom_classifier.fit(data_train, target_train)
custom_predictions = custom_classifier.predict(data_test)
    
custom_accuracy = np.mean(custom_predictions == target_test) * 100

"""
COMPARING TO SKLEARN KNN
"""
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(data_train, target_train)
predictions = classifier.predict(data_test)

sklearn_accuracy = np.mean(predictions == target_test) * 100

"""
I only need to know if my algorithm works equally as accurately as sklearn's
"""
if (custom_accuracy == sklearn_accuracy):
    print(str(custom_accuracy) + " = " + str(sklearn_accuracy) + "!\nSuccess!")
else:
    print(str(custom_accuracy) + " != " + str(sklearn_accuracy))
