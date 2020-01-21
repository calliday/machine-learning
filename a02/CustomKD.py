#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:59:14 2020

@author: Caleb

My custom class that uses the KNN method of machine learning, but also
implements a KD tree

I used much advice and code from Tsoding, a smart Russian programmer on YouTube
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
The recursive meat of building the KD tree
"""
def build_kd(points, k=1, depth=0):
    n = len(points)
        
    if n == 0:
        return None
        
    axis = depth % k
    
    sorted_points = sorted(points, key = lambda point: point[axis])
    
    # Can't use a float, so making it an int
    n_over_2 = int(n/2)
    
    # Assuring that there are enough points in each leaf of the tree
    # for the chosen k
    if np.where(sorted_points[n_over_2]) < k:
        return
    
    # The final recursive call that builds the tree
    return {
        'point': sorted_points[n_over_2],
        'left': build_kd(sorted_points[n_over_2], k, depth + 1),
        'right': build_kd(sorted_points[n_over_2 + 1], k, depth + 1)
    }
    
"""
Gets the closest kd tree leaf to a point
"""
def get_kd_leaf(root, point, k, depth=0, best=None):
    if root is None:
        return best
    
    axis = depth % k
    
    next_best = None
    next_branch = None
    
    if best is None or compute_distance(point, best) > compute_distance(point, root['point']):
        next_best = root['point']
    else:
        next_best = best
        
    if point[axis] < root['point'][axis]:
        next_branch = root['left']
    else:
        next_branch = root['right']
    
    return get_kd_leaf(next_branch, point, k, depth+1, next_best)

"""
CUSTOM K NEIGHBORS CLASS

My custom class that implements the KNN algorithm structure
"""
class CustomKNeighbors:
    def __init__(self, k=1):
        if k < 0:
            self.k = 1
        else:
            self.k = k
        self.data_train = np.array
        self.target_train = np.array
            
    # Fit the data into a K-D Tree
    def fit(self, data_train=np.array, target_train=np.array):
        self.data_train = build_kd(data_train, self.k)
        self.target_train = target_train
    
    def predict(self, data_test):
        distances = []
        
        """ probably will have to add searching for the right branch here """
        print(get_kd_leaf(self.data_train, data_test[0], self.k))
        return np.array([0])
    
        for i, test in enumerate(data_test):
            distances.append([])
            for train in self.data_train:
                distances[i].append(compute_distance(test, train))
        
        # Transfer the closest distances into an array of sorted closest distance indexes
        knn_indexes = []
        for distance in distances:
            knn_indexes.append(np.argsort(distance))
            
        """ probably will have to change this """
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
        
