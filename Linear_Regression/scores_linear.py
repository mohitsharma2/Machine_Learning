# Now if I want to get 100 scores 
# How much hours do i need to study 

# Reverse the features and labels and try the model and prediction again
# You might need to reshape the features by features = featurtes.reshape(25,1)
# Now regressor.predict(100)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/student_scores.csv')

print (dataset.shape)
print (dataset.ndim)
print (dataset.head())
print (dataset.describe())

print (dataset.dtypes)

#Check if any NaN values in dataset
dataset.isnull().any(axis=0)

# Check for outlier values
# Also helps in finding the data distribution 
plt.boxplot(dataset.values)
                        
features =dataset.iloc[:, -1].values 
labels = dataset.iloc[:, :1].values
reshape=features.reshape(25,1)
from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  

regressor.fit(reshape, labels) 

print (regressor.predict([[100]])) # [100.24170731]  this is a label
import matplotlib.pyplot as plt
plt.axis([10,110,1,10])

plt.scatter(reshape, labels, color = 'brown')
plt.plot(features, regressor.predict(reshape), color = 'olive')
plt.title('Study Hours and Exam Score')
plt.xlabel('Exam Score: Marks')
plt.ylabel('Study Hours')
plt.show()






































