# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# BEGIN above and beyond attempting to read the data from a file

f = open("iris.data", "r")

flowers = []
data = []

for line in f:
    parts = line.split(',')
    flowers.append(parts[-1:][-2:]) # the [-2:] is made to get rid of '\n'
    data.append(parts[:4])

# print(flowers)
# print(data)
        
# END file reading testing
    
    
# Instead of the file reading, for now I will use sklearn
# Setting up the training and testing data
iris = datasets.load_iris()
    
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, train_size=0.3)


# Training and testing the data
classifier = GaussianNB()
classifier.fit(data_train, target_train)

targets_predicted = classifier.predict(data_test)

# Print error if one flower was incorrectly classified
# In other words, display accuracy

count = 0
correct_count = 0

for i, prediction in enumerate(targets_predicted):
    count += 1
    if prediction == target_test[i]:
        correct_count += 1
        
print("\nGaussianNB accuracy: " + str(correct_count / count * 100) + "%\n")


"""
HARD CODED CLASSIFIER
My custom classifier
"""
class HardCodedClassifier:
    
    def fit(self, data = []):
        print("Data successfully fitted.\n")
        return
    
    def predict(self, data):
        
        predicted_list = []
        
        for item in data:
            predicted_list.append(0)
            
        return predicted_list
    
    def display_accuracy(self, data_test, target_test):
        targets_predicted = self.predict(data_test)
        
        count = 0
        correct_count = 0

        for i, prediction in enumerate(targets_predicted):
            count += 1
            if prediction == target_test[i]:
                correct_count += 1
                
        print("HardCodedClassifier accuracy: " + str(correct_count / count * 100) + "%\n")
    
"""END CLASS"""
          
# Testing out the custom classifier  
custom_classifier = HardCodedClassifier()

custom_classifier.fit(data_train)

prediction = custom_classifier.predict(data_test)

custom_classifier.display_accuracy(data_test, target_test)
