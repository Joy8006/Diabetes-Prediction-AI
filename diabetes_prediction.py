import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm



# Reading the data file
diabetes_dataset = pd.read_csv('diabetes.csv')


# Separating data and labels
x = diabetes_dataset.drop(columns='Outcome', axis= 1)
y = diabetes_dataset['Outcome']

standard_data = StandardScaler().fit_transform(x)

X = standard_data
Y = y

X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)


# Calling the model
classifier = svm.SVC(kernel='linear')

# Tranining the model
classifier.fit(X_train, Y_train)

# Testing Accuracy score of the training data

X_train_accuracy = classifier.predict(X_train)

train_accuracy = accuracy_score(X_train_accuracy, Y_train)

print("Accuracy score of the training data is {} %".format(train_accuracy*100))

#  Testing Accuracy score of the test data
X_test_accuracy = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_accuracy, Y_test)


print("Accuracy score of the testing data is {} %".format(test_accuracy*100))

# # Making it user friendly and taking data from USER
# prompt1 = "Pregnancies"
# prompt2 = "Glucose"
# prompt3 = "BloodPressure"
# prompt4 = "SkinThickness"
# prompt5 = "Insulin"
# prompt6 = "BMI"
# prompt7 = "DiabetesPedigreeFunction"
# prompt8 = "Age"
# d1 = float(input(prompt1))
# d2 = float(input(prompt2))
# d3 = float(input(prompt3))
# d4 = float(input(prompt4))
# d5 = float(input(prompt5))
# d6 = float(input(prompt6))
# d7 = float(input(prompt7))
# d8 = float(input(prompt8))

# input_data = (d1,d2,d3,d4,d5,d6,d7,d8)

# input_data_as_numpy_array = np.asarray(input_data)

# input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# prediction = classifier.predict(input_data_reshaped)

# if prediction==0:
#     print("You are not Diabetic")

# elif prediction ==1:
#     print("You are Diabetic")
