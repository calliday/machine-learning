#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:04:23 2020

@author: Caleb
"""

import numpy as np

""" If a numpy array dataset is continuous float values, make them discrete """
def processData(data, split):
    # Get mins and maxs for each column
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    
    # Normalize everything to 3
    for i, row in enumerate(data):
        for j, column in enumerate(row):
            data[i][j] -= mins[j]
            temp = int(data[i][j]/((maxs[j]-mins[j])/split))
            if temp == split:
                temp = split - 1
            data[i][j] = temp

""" Returns the entropy value of a 1-dimensional array """
def getEntropy(one_d_array):
    entropy = 0
    # Grab count of targets in the array
    bins = np.bincount(one_d_array)
    total = np.sum(bins)
    
    # Total the entropy of the whole set
    for count in bins:
        if count != 0:
            entropy += (count/total) * (np.log2(count/total))
    
    # Return it
    return -entropy

""" A nice and condensed way to get the most common target """
def mostCommonValue(npArray):
    return np.argmax(np.bincount(npArray))

""" The recursive function to get the entire tree """
def getTree(data, targets, remFeatures):
    # If all examples have same label, return node with that label
    if all(target for target in targets == targets[0]):
        return DTCNode(targets[0])
    
    # If there are no more features to test, return leaf node with most common value 
    if np.size(remFeatures) == 0:
        return DTCNode(mostCommonValue(targets))
    
    # Find the best entropy of the remaining feature splits
    entropies = []
    for feature in remFeatures:
        # Grab only the column needed for the feature
        column = data[:,feature]
        # The possible values within the column
        possibilities = np.unique(column)
        total = np.size(column)
        # Loop through the possible values to create a pool for each entropy test
        featureEntropy = 0
        for possibility in possibilities:
            # Only check the target pool to total entropy
            target_pool = targets[np.where(data[:,feature] == possibility)]
            featureEntropy += (np.size(target_pool)/total) * getEntropy(target_pool)
        
        entropies.append(featureEntropy)
    
    # Create a node with the best entropy
    newNode = DTCNode()
    newNode.feature = remFeatures[np.argmin(entropies)]
    
    # Build the tree down from there
    # Loop through remaining features
    i = newNode.feature
    # Grab only the column needed (based on the feature)
    column = data[:,i]
    # The unique possible values within the column
    possibilities = np.unique(column)
    newNode.possibilities = possibilities
    
    # Loop through the possible values to create a pool for each branch
    for j, possibility in enumerate(possibilities):
        # Create the separated pools to pass to each branch
        data_pool = data[:,:][np.where(data[:,i] == possibility)]
        target_pool = targets[np.where(data[:,i] == possibility)]
        # For each value in the column of concern
        if len(remFeatures) > 1:
            remFeatures = np.delete(remFeatures, np.where(remFeatures == i), 0)
        else:
            remFeatures = []
        newNode.children.append(getTree(data_pool, target_pool, remFeatures))
    
    return newNode

""" Recursively find the appropriate leaf node """
def find(node, test):
    # The end case: if the node is a leaf
    if node.target != None:
        return node.target
    
    # Grab the appropriate index of the branch to traverse down
    possibilities = np.array(node.possibilities)
    index = np.where(possibilities==test[node.feature])
    
    # If there was no possible result, return a default of target 0
    if (len(index[0]) < 1):
        return find(node.children[0], test)
    
    # Move down the next branch
    return find(node.children[index[0][0]], test)

""" The node that can become a tree """
class DTCNode:
    def __init__(self, target=None):
        self.children = []
        self.possibilities = None # Keeps track of the index of each branch (based on available feature values)
        self.feature = None # Keeps track of feature that this node is comparing
        self.target = target # This the value of a leaf or a premature leaf value

""" The big ID3 classifier """
class ID3Classifier:
    
    # Initialize the root to be empty
    def __init__(self):
        self.root = None

    # Get the decision tree by sending the data and root node through getTree
    def fit(self, data, targets):
        self.root = getTree(data, targets, np.arange(np.size(data, 1)))
    
    # Return the predictions
    def predict(self, test_data):
        results = []
        for test in test_data:
            results.append(find(self.root, test))
        return results
