"""
Q1. (Create a program that fulfills the following specification.)
PastHires.csv


Here, we are building a decision tree to check if a person is hired or not 
based on certain predictors.

Import PastHires.csv File.

scikit-learn needs everything to be numerical for decision trees to work.

So, use any technique to map Y,N to 1,0 and levels of education to some scale of 0-2.

    Build and perform Decision tree based on the predictors and see how accurate your prediction is for a being hired.

Now use a random forest of 10 decision trees to predict employment of specific candidate profiles:

    Predict employment of a currently employed 10-year veteran, previous employers 4, went to top-tire school, having Bachelor's Degree without Internship.
    Predict employment of an unemployed 10-year veteran, ,previous employers 4, didn't went to any top-tire school, having Master's Degree with Internship.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

dataset = pd.read_csv("C:/Users/mohit/Desktop/Machine_Learning/Classification/PastHires.csv")  

dataset.head()
pd.set_option('display.max_columns', None)
dataset.head()

# Finding missing data
dataset.isnull().any(axis=0)
dataset.columns

features = dataset.drop('Hired', axis=1)
print(features)
print(features.shape)

  
labels = dataset['Hired']  
print(labels)
print(labels.shape)


from sklearn.model_selection import train_test_split  

features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.20)  


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()

features[:,1] = labelencoder.fit_transform(features[:,1])

features[:,3] = labelencoder.fit_transform(features[:,3])

features[:,4] = labelencoder.fit_transform(features[:,4])

features[:,5] = labelencoder.fit_transform(features[:,5])

features[:,6] = labelencoder.fit_transform(features[:,6])



# One Hot Encoding 
onehotencoder = OneHotEncoder(categorical_features = [1])
features = onehotencoder.fit_transform(features).toarray()

onehotencoder = OneHotEncoder(categorical_features = [3])
features = onehotencoder.fit_transform(features).toarray()

onehotencoder = OneHotEncoder(categorical_features = [4])
features = onehotencoder.fit_transform(features).toarray()

onehotencoder = OneHotEncoder(categorical_features = [5])
features = onehotencoder.fit_transform(features).toarray()

onehotencoder = OneHotEncoder(categorical_features = [6])
features = onehotencoder.fit_transform(features).toarray()
print(features)





from sklearn.tree import DecisionTreeClassifier  

classifier = DecisionTreeClassifier()  

classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test) 

my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})

print(my_frame)


