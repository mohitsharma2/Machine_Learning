"""
Code Challenge
  Name: 
    Box Office Collection Prediction Tool
  Filename: 
    Bahubali2vsDangal.py
  Dataset:
    Bahubali2vsDangal.csv
  Problem Statement:
    It contains Data of Day wise collections of the movies Bahubali 2 and Dangal 
    (in crores) for the first 9 days.
    
    Now, you have to write a python code to predict which movie would collect 
    more on the 10th day.
  Hint:
    First Approach - Create two models, one for Bahubali and another for Dangal
    Second Approach - Create one model with two labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Bahubali2_vs_Dangal.csv")

#    First Approach - Create two models, one for Bahubali and another for Dangal

features=df.iloc[:,0].values
labels=df.iloc[:,1].values
features=features.reshape(9,1)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features,labels)

print(regressor.predict([[10]]))

plt.scatter(features, labels, color = 'olive')
plt.plot(features, regressor.predict(features), color = 'red')
plt.title('Bahubali2-Day collection')
plt.xlabel('Day')
plt.ylabel('Collection')
plt.show()


"""
output==============================================================

[17.41666667]

"""


features=df.iloc[:,0].values
labels=df.iloc[:,-1].values
features=features.reshape(9,1)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features,labels)

print(regressor.predict([[10]]))


plt.scatter(features, labels, color = 'olive')
plt.plot(features, regressor.predict(features), color = 'green')
plt.title('Dangal-Day collection')
plt.xlabel('Day')
plt.ylabel('Collection')
plt.show()

"""
output===>
[19.25944444]

"""

#    Second Approach - Create one model with two labels


df=pd.read_csv("Bahubali2_vs_Dangal.csv")

features=df.iloc[:,[0]].values
print(type(features))
print(features.shape)


labels=df.iloc[:,1:].values
print(type(labels))
print(labels.shape)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features,labels)

print(regressor.predict([[10]]))

"""
output===>

[[17.41666667 19.25944444]]

"""
