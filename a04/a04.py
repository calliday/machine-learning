#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:17:58 2020

@author: Caleb
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier   # Sklearn's ID3 classifier
import DecisionTreeClassifier as dtc              # My custom ID3 classifier
from customClassifier import CustomKNeighbors     # My old KNN Classifier

# Get the data
iris = datasets.load_iris()
targets = np.array(iris.target)
data = np.array(iris.data)

# Preprocessing the data
dtc.processData(data, 3)

data_train, data_test, target_train, target_test = train_test_split(data, targets, train_size=0.3)

# Test my tree
ID3 = dtc.ID3Classifier()
ID3.fit(data_train, target_train)
mine = ID3.predict(data_test)

# Test sklearn's tree
realID3 = DecisionTreeClassifier()
realID3.fit(data_train, target_train)
theirs = realID3.predict(data_test)

# Test KNN tree
old = CustomKNeighbors(3)
old.fit(data_train, target_train)
old = old.predict(data_test)

# Output the results
print("My tree vs sklearn's tree: ")
print(np.mean(mine == target_test) * 100)
print(np.mean(theirs == target_test) * 100)
print()
print("My tree vs my old KNN")
print(np.mean(mine == target_test) * 100)
print(np.mean(old == target_test) * 100)