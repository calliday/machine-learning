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

# print(flowers) # can uncomment if I want to display the results of file reading
# print(data)    # ditto
        
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
for i, prediction in enumerate(targets_predicted):
    if prediction != target_test[i]:
        print("Incorrect prediction!")

"""
HARD CODED CLASSIFIER
My custom classifier
"""
class HardCodedClassifier:
    
    def fit(self, data = []):
        print("Data successfully fitted.")
        return
    
    def predict(self, data):
        
        predicted_list = []
        
        for item in data:
            predicted_list.append(0)
            
        return predicted_list
    
"""END CLASS"""
          
# Testing out the custom classifier  
custom_classifier = HardCodedClassifier()

custom_classifier.fit(data_train)

prediction = custom_classifier.predict(data_test)

# Success!
print(prediction)
