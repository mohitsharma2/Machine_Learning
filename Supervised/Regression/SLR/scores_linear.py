# Now if I want to get 100 scores 
# How much hours do i need to study 

#Hint

# Reverse the features and labels and try the model and prediction again
# You might need to reshape the features by features = featurtes.reshape(25,1)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')

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
                        
features =dataset.iloc[:,-1].values 
print(type(features))
print(features.shape)
features=features.reshape(25,1)


labels = dataset.iloc[:, :1].values
print(type(labels))
print(labels.shape)



#or we can write  

#features =dataset.iloc[:,[1]].values 


from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  

regressor.fit(features, labels) 

print (regressor.predict([[100]])) # [[9.74174391]]  this is a label


plt.scatter(features, labels, color = 'brown')
plt.plot(features, regressor.predict(features), color = 'olive')
plt.title('Study Hours and Exam Score')
plt.xlabel('Exam Score: Marks')
plt.ylabel('Study Hours')
plt.show()
