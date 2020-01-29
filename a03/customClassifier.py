#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:27:32 2020

@author: Caleb

My custom class that uses the KNN method of machine learning
"""

import numpy as np
from collections import Counter

"""
A somewhat private function that computes the distance of two points
"""
def compute_distance(test, train):
    test = np.array(test)
    train = np.array(train)
    return np.sqrt(np.sum((test-train)**2))


"""
CUSTOM K NEIGHBORS CLASS

My custom class that implements the KNN algorithm structure
"""
class CustomKNeighbors:
    def __init__(self, k=0):
        if k < 0:
            self.k = 1
        else:
            self.k = k
        self.data_train = np.array
        self.target_train = np.array
            
    def fit(self, data_train=np.array, target_train=np.array):
        self.data_train = data_train
        self.target_train = target_train
    
    def predict(self, data_test):
        distances = []
    
        for i, test in enumerate(data_test):
            distances.append([])
            for train in self.data_train:
                distances[i].append(compute_distance(test, train))
        
        # Transfer the closest distances into an array of sorted closest distance indexes
        knn_indexes = []
    
        for distance in distances:
            knn_indexes.append(np.argsort(distance))
            
        # 2d array for each of the 'k' closest distance 'trained' indexes
        custom_closest = []
        for i, ordered in enumerate(knn_indexes):
            custom_closest.append([])
            for closest in ordered[:self.k]:
                custom_closest[i].append(self.target_train[closest])
                
        # Find the most common target of the closest 'k' points
        prediction = []
        
        for row in custom_closest:
            prediction.append(Counter(row).most_common(1)[0][0])
        
        return np.array(prediction)
    
    def get_accuracy(self, data_test, target_test):
        prediction = self.predict(data_test)
        return np.mean(prediction != target_test) * 100
        