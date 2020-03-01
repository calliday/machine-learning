#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:57:06 2020

@author: Caleb
"""

import numpy as np
from random import seed
import time
from random import random
import math

''' Get random values between -1 and 1 '''
def getRandom(size=1):
    seed(time.time() % 1)
    smalls = np.zeros(size)
    for i in range(size):
    	smalls[i] = random()
    return (smalls * 2 - 1)
    
''' Process numerical data into cagetorical measurements '''
def normalizeData(data):
    # Get mins and maxs for each column
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    
    # Normalize everything to 3
    for i, row in enumerate(data):
        for j, column in enumerate(row):
            data[i][j] -= mins[j]
            data[i][j] = data[i][j]/(maxs[j]-mins[j])
            
''' The basic implementation of a neural network '''
class BasicNN:
    
    ''' Setting class data '''
    def __init__(self):
        self.data = None
        self.targets = None
        self.layers = []
        self.numLayers = None
        self.biasValue = None
        return

    ''' Set data, targets, and layer specific information '''
    def fit(self, data=[], targets=[], hiddenLayersShape=[], biasValue=-1):
        # Set class data
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.biasValue = biasValue
        self.numLayers = len(hiddenLayersShape)
        
        # Will keep track of number of previous layer's nodes
        prevNumNodes = 0
        # Set up each HIDDEN layer
        for i in range(0, self.numLayers):
            # If it is the first layer
            if not i:
                # Grab num "Nodes" of each data example
                prevNumNodes = data.shape[1] + 1
            # It is not the first layer
            else:
                # Grab numNodes of previous layer
                prevNumNodes = self.layers[i - 1].numNodes + 1
            tempLayer = self.Layer(hiddenLayersShape[i])
            tempLayer.weights = getRandom(prevNumNodes)[np.newaxis]
            for _ in range(1, tempLayer.numNodes):
                tempLayer.weights = np.append(tempLayer.weights, getRandom(prevNumNodes)[np.newaxis], 0)
            # Append the layer
            self.layers.append(tempLayer)
                    
        # The OUTPUT layer should have as many nodes as unique target values there are
        tempLayer = self.Layer(len(np.unique(self.targets)))
        # If there are hidden layers...
        if self.numLayers:
            # ...get the numNodes of the last one to generate weights
            prevNumNodes = self.layers[self.numLayers - 1].numNodes + 1
        # There are no hidden layers...
        else:
            # ... so get the num "Nodes" of each data example
            prevNumNodes = data.shape[1] + 1
        # Make sure the starting weights size is appropriate
        tempLayer.weights = getRandom(prevNumNodes)[np.newaxis]
        # Append the proper amount of weights
        for _ in range(1, tempLayer.numNodes):
            tempLayer.weights = np.append(tempLayer.weights, getRandom(prevNumNodes)[np.newaxis], 0)
        self.layers.append(tempLayer)
        self.numLayers += 1
        
        return
    
    ''' The backpropagation of the network, it also forward feeds each example iteration '''
    def train(self, learningRate=0.2, epochs=100):
        
        for epoch in range(0, epochs):
            # Temporarily make biased data
            biasedData = np.append((np.full((self.data.shape[0], 1), self.biasValue)), self.data, 1)
            for i, example in enumerate(biasedData):
                self.getOutputs(example) # Don't need the outputs here, but this forward propogates
                target = self.targets[i]
                for tempIndex, layer in enumerate(reversed(self.layers)):
                    j = len(self.layers) - 1 - tempIndex
                    
                    # Get a reference to the left layer
                    left = None
                    if j != 0: # Not the first layer
                        left = self.layers[j - 1]
                        av = np.append(-1, left.activatedValues)
                            
                    for nodei in range(0, layer.numNodes):
                        # The expected value from each output node by default is 0
                        a = layer.activatedValues[nodei]
                        t = 0
                        error = 0
                        # But it is 1 if it is the target node
                        if nodei == target:
                            t = 1
                        # Find the error for the individual node
                        # The layer is the output layer
                        if j == self.numLayers - 1:
                            error = a * (1 - a) * (a - t)
                        # The layer is a hidden layer
                        else:
                            sumOfErrors = 0
                            right = self.layers[j + 1]
                            for rnode in range(0, right.numNodes):
                                sumOfErrors += right.weights[rnode][nodei] * right.errors[rnode]
                            error = a * (1 - a) * sumOfErrors
                        
                        # Append the error to the layer for the layers to the right
                        layer.errors = np.append(layer.errors, error)
                            
                        for weighti in range(0, len(layer.weights[nodei])):
                            ai = None
                            if left != None:
                                ai = av[weighti]
                            else:
                                ai = example[weighti]

                            layer.weights[nodei][weighti] = layer.weights[nodei][weighti] - (learningRate * error * ai)
        
        return
    
    ''' The batch version of training. The default is a batch size of 5 '''
    def batchTrain(self, learningRate=0.2, epochs=100, batch_size=5):
        
        for epoch in range(0, epochs):
            # Temporarily make biased data
            biasedData = np.append((np.full((self.data.shape[0], 1), self.biasValue)), self.data, 1)
            for i, example in enumerate(biasedData):
                self.getBatchOutputs(example) # Don't need the outputs here, but this forward propogates
                batchFlag = False
                if i % batch_size == 0:
                    for layer in self.layers:
                        layer.resetBatch()
                        if i != 0:
                            batchFlag = True
                            layer.crunchBatch(5)
                target = self.targets[i]
                for tempIndex, layer in enumerate(reversed(self.layers)):
                    j = len(self.layers) - 1 - tempIndex
                    
                    # Get a reference to the left layer
                    left = None
                    if j != 0: # Not the first layer
                        left = self.layers[j - 1]
                        if not batchFlag:
                            av = np.append(-1, left.activatedValues)
                        else:
                            av = np.append(-1, left.batchActivated)
                            
                    for nodei in range(0, layer.numNodes):
                        # The expected value from each output node by default is 0
                        if not batchFlag:
                            a = layer.activatedValues[nodei]
                        else:
                            a = layer.batchActivated[nodei]
                        t = 0
                        error = 0
                        # But it is 1 if it is the target node
                        if nodei == target:
                            t = 1
                        # Find the error for the individual node
                        # The layer is the output layer
                        if j == self.numLayers - 1:
                            error = a * (1 - a) * (a - t)
                        # The layer is a hidden layer
                        else:
                            sumOfErrors = 0
                            right = self.layers[j + 1]
                            for rnode in range(0, right.numNodes):
                                if not batchFlag:
                                    deltak = right.errors[rnode]
                                else:
                                    deltak = right.batchErrors[rnode]
                                sumOfErrors += right.weights[rnode][nodei] * deltak
                            error = a * (1 - a) * sumOfErrors
                        
                        # Append the error to the layer for the layers to the right
                        layer.errors = np.append(layer.errors, error)
                        
                        # Only update the weights if we reached the batch size
                        if batchFlag:
                            for weighti in range(0, len(layer.weights[nodei])):
                                ai = None
                                if left != None:
                                    ai = av[weighti]
                                else:
                                    ai = example[weighti]
    
                                layer.weights[nodei][weighti] = layer.weights[nodei][weighti] - (learningRate * error * ai)
                    layer.collectBatchErrors()
        
        return
    
    ''' The testing phase '''
    def predict(self, data):
        
        predictions = []
        
        # In each layer's turn, temporarily make biased data
        biasedData = np.append((np.full((data.shape[0], 1), self.biasValue)), data, 1)
        for i, example in enumerate(biasedData):
            prediction = 0
            outputs = self.getOutputs(example)

            # The firing node is the node with the max activation value
            prediction = np.argmax(outputs)
            ''' # Old way of determing firing node
            for j, value in enumerate(outputs):
                if value > 0.5:
                    prediction = j
            '''
                    
            predictions.append(prediction)
        
        return np.array(predictions)
    
    ''' 1 forward propagation of a data example through the network '''
    def getOutputs(self, example=None):
        # In each layer's turn, temporarily make biased data
        self.resetLayerValues()
        for layerCount in range(0, self.numLayers):
            currentLayer = self.layers[layerCount]

            if layerCount: # If we are at anything but first layer
                inputValues = np.append(self.biasValue, self.layers[layerCount - 1].activatedValues)
                # Now that we have our inputs, alter the current layer's values
                for currentNode in range(0, currentLayer.numNodes):
                    currentLayer.values = np.append(currentLayer.values, np.matmul(inputValues, currentLayer.weights[currentNode][np.newaxis].T))
                
            else: # we are at the first layer
                for currentNode in range(0, currentLayer.numNodes):
                    currentLayer.values = np.append(currentLayer.values, np.matmul(example, currentLayer.weights[currentNode][np.newaxis].T))
            
            # Set the layer's activated values
            currentLayer.activate()        
            
        # Put the prediction in a list
        outputs = self.layers[self.numLayers - 1].activatedValues
        
        return outputs
    
    ''' The forward propagation of a batch of examples is slightly different '''
    def getBatchOutputs(self, example=None):
        # In each layer's turn, temporarily make biased data
        self.resetLayerValues()
        for layerCount in range(0, self.numLayers):
            currentLayer = self.layers[layerCount]

            if layerCount: # If we are at anything but first layer
                inputValues = np.append(self.biasValue, self.layers[layerCount - 1].activatedValues)
                # Now that we have our inputs, alter the current layer's values
                for currentNode in range(0, currentLayer.numNodes):
                    currentLayer.values = np.append(currentLayer.values, np.matmul(inputValues, currentLayer.weights[currentNode][np.newaxis].T))
                
            else: # we are at the first layer
                for currentNode in range(0, currentLayer.numNodes):
                    currentLayer.values = np.append(currentLayer.values, np.matmul(example, currentLayer.weights[currentNode][np.newaxis].T))
            
            # Set the layer's activated values
            currentLayer.activate()
            # We are also setting batch values
            currentLayer.collectBatchActivated()
            
        # Put the prediction in a list
        outputs = self.layers[self.numLayers - 1].activatedValues
        
        return outputs 
    
    ''' Reset the values set in forward propogation '''
    def resetLayerValues(self):
        for layer in self.layers:
            layer.reset()
    
    ''' A class to hold node and neural network information for one layer '''
    class Layer:
        def __init__(self, numNodes=1):
            self.numNodes = numNodes
            self.weights = np.array([[]])
            self.values = np.array([])
            self.activatedValues = np.array([])
            self.errors = np.array([])
            # For batch training
            self.batchActivated = np.array([])
            self.batchErrors = np.array([])
            return
        
        ''' Convert the node values using an activation function '''
        def activate(self):
            for value in self.values:
                self.activatedValues = np.append(self.activatedValues, 1 / (1 + (math.exp(-value))))
            return
        
        ''' Reset the layer values that are used in forward propogation '''
        def reset(self):
            self.values = np.array([])
            self.activatedValues = np.array([])
            self.errors = np.array([])
            return
        
        ''' Adds up the batch values '''
        def collectBatchActivated(self):
            if self.batchActivated.shape != self.activatedValues.shape:
                self.resetBatch()
            self.batchActivated += self.activatedValues
            return
        
        ''' Add the node errors to the batch errors, to be averaged later '''
        def collectBatchErrors(self):
            if self.batchErrors.shape != self.errors.shape:
                self.resetBatch()
            self.batchErrors += self.errors
            return
            
        ''' Average the batch activations and errors '''
        def crunchBatch(self, batch_size=5):
            self.batchActivated = self.batchActivated / batch_size
            self.batchErrors = self.batchErrors / batch_size
            return
            
        ''' Reset the batch values at each batch iteration '''
        def resetBatch(self):
            self.batchActivated = np.zeros(self.activatedValues.shape)
            self.batchErrors = np.zeros(self.errors.shape)
            return
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        