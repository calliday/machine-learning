import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
import numpy as np
import sys

# Above and Beyond
from customClassifier import CustomKNeighbors

# Display formatting
pd.options.display.max_columns = 999
pd.options.display.max_rows = 5

np.set_printoptions(threshold=sys.maxsize)


""" LOAD each of the files """

# Actual loading
pd_student = pd.read_csv("data/student-mat.csv", sep=";")
pd_mpg = pd.read_csv("data/auto-mpg.data", header=None, delim_whitespace=True)
pd_car = pd.read_csv("data/car.data", sep=",")


""" 
    STUDENT
            """
""" PRE-PROCESS """

# Label encode everything
pd_student.school = pd_student.school.astype("category").cat.codes
pd_student.sex = pd_student.sex.astype("category").cat.codes
pd_student.age = pd_student.age.astype("category").cat.codes
pd_student.address = pd_student.address.astype("category").cat.codes
pd_student.famsize = pd_student.famsize.astype("category").cat.codes
pd_student.Pstatus = pd_student.Pstatus.astype("category").cat.codes
pd_student.Mjob = pd_student.Mjob.astype("category").cat.codes
pd_student.Fjob = pd_student.Fjob.astype("category").cat.codes
pd_student.reason = pd_student.reason.astype("category").cat.codes
pd_student.guardian = pd_student.guardian.astype("category").cat.codes
pd_student.schoolsup = pd_student.schoolsup.astype("category").cat.codes
pd_student.famsup = pd_student.famsup.astype("category").cat.codes
pd_student.paid = pd_student.paid.astype("category").cat.codes
pd_student.activities = pd_student.activities.astype("category").cat.codes
pd_student.nursery = pd_student.nursery.astype("category").cat.codes
pd_student.higher = pd_student.higher.astype("category").cat.codes
pd_student.internet = pd_student.internet.astype("category").cat.codes
pd_student.romantic = pd_student.romantic.astype("category").cat.codes

# Generate numpy arrays
target_student = np.array(pd_student["G3"])[1:] # Only the targets
data_student = np.array(pd_student.drop("G3", axis=1).drop([0], axis=0)) # Remove the labels

# Standardize everything
#data_student_scaled = preprocessing.StandardScaler().fit(data_student)
#data_student = data_student_scaled.transform(data_student)

""" TRAIN through KNNRegressor """
# Split data up
st_data_train, st_data_test, st_target_train, st_target_test = train_test_split(
        data_student, target_student, train_size=0.3)

# Train and grab predictions
regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(st_data_train, st_target_train)
st_predictions = regr.predict(st_data_test)

# Get the mean error of each value
st_error = np.sqrt(metrics.mean_squared_error(st_predictions, st_target_test))

# (st_target_test - st_error) < predictions < (st_target_test + st_error)
st_low = (st_target_test - st_error) < st_predictions
st_high = (st_target_test + st_error) > st_predictions
st_binary = (st_low == st_high).astype(int)
print("\nStudent grade accuracy:\n((TP(+/- stderr) + TN(+/- stderr)) / total) = " +
      str(np.mean(st_binary) * 100) + "%\n")




""" 
    MPG
        """
""" PRE-PROCESS """
# Prepare data and targets
pd_mpg = np.array(pd_mpg)[:,0:-1] # Remove the last row
target_mpg = pd_mpg[:,:1] # The first row is the targets
data_mpg = pd_mpg[:,1:] # The rest is the data

# Standardize everything
data_mpg_scaled = preprocessing.StandardScaler().fit(data_mpg)
data_mpg = data_mpg_scaled.transform(data_mpg)            

""" TRAIN through KNNClassifier """
# Split data up
mpg_data_train, mpg_data_test, mpg_target_train, mpg_target_test = train_test_split(
        data_mpg, target_mpg, train_size=0.3)

# Train and grab predictions
mpg_regressor = KNeighborsRegressor(n_neighbors=3)
mpg_regressor.fit(mpg_data_train, mpg_target_train)
mpg_predictions = mpg_regressor.predict(mpg_data_test)

# Get the mean error of each value
mpg_error = np.sqrt(metrics.mean_squared_error(mpg_predictions, mpg_target_test))

# (mpg_target_test - mpg_error) < predictions < (mpg_target_test + mpg_error)
mpg_low = (mpg_target_test - mpg_error) < mpg_predictions
mpg_high = (mpg_target_test + mpg_error) > mpg_predictions
mpg_binary = (mpg_low == mpg_high).astype(int)
print("MPG accuracy:\n((TP(+/- stderr) + TN(+/- stderr)) / total) = " +
      str(np.mean(mpg_binary) * 100) + "%\n")




""" 
     CARS
            """
""" PRE-PROCESS """
# Convert labels to numerical values   
car_discrete_values = {
            "buying":   { "vhigh":4, "high":3, "med":2, "low":1     },
            "maint":    { "vhigh":4, "high":3, "med":2, "low":1     },
            "doors":    { "2":2, "3":3, "4":4, "5more":5            },
            "persons":  { "2":1, "4":2, "more":4 },
            "lug_boot": { "small":1, "med":2, "big":3               },
            "safety":   { "low":1, "med":2, "high":3                },
            "target":   { "unacc":0, "acc":1, "good":2, "vgood":3   }
        }

pd_car.replace(car_discrete_values, inplace=True)

# Prepare data and targets
pd_car = np.array(pd_car)[1:,:] # Convert to np.array and remove headers
target_car = pd_car[:,-1:] # The last row is the targets
data_car = pd_car[:,:-1] # The rest is the data

# Standardize everything
data_car_scaled = preprocessing.StandardScaler().fit(data_car)
data_car = data_car_scaled.transform(data_car)

""" TRAIN through KNNClassifier """
# Split data up
car_data_train, car_data_test, car_target_train, car_target_test = train_test_split(
        data_car, target_car, train_size=0.3)

# Train and grab predictions
car_classifier = KNeighborsClassifier(n_neighbors=3)
car_classifier.fit(car_data_train, car_target_train.ravel())
car_predictions = car_classifier.predict(car_data_test)

print("Car acceptibility accuracy:\n((TP + TN / total) = " +
      str(np.mean(car_predictions == car_target_test) * 100) + "%\n")


""" A&B: Implementing Custom Classifier"""
car_custom_classifier = CustomKNeighbors(3)
car_custom_classifier.fit(car_data_train, car_target_train.ravel())
custom_predictions = car_custom_classifier.predict(car_data_test)

print("Car acceptibility accuracy (from custom classifier):\n((TP + TN / total) = " +
      str(np.mean(custom_predictions == car_target_test) * 100) + "%\n")