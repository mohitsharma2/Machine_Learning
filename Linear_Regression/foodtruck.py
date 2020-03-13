"""
Code Challenge
  Name: 
    Food Truck Profit Prediction Tool
  Filename: 
    Foodtruck.py
  Dataset:
    Foodtruck.csv
  Problem Statement:
    Suppose you are the CEO of a restaurant franchise and are considering 
    different cities for opening a new outlet. 
    
    The chain already has food-trucks in various cities and you have data for profits 
    and populations from the cities. 
    
    You would like to use this data to help you select which city to expand to next.
    
    Perform Simple Linear regression to predict the profit based on the 
    population observed and visualize the result.
    
    Based on the above trained results, what will be your estimated profit, 
    
    If you set up your outlet in Jaipur? 
    (Current population in Jaipur is 3.073 million)
        
  Hint: 
    You will implement linear regression to predict the profits for a 
    food chain company.
    Foodtruck.csv contains the dataset for our linear regression problem. 
    The first column is the population of a city and the second column is the 
    profit of a food truck in that city. 
    A negative value for profit indicates a loss.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/foodtruck.csv")
#print(df)

features=df.iloc[:,:-1]
labels=df.iloc[:,1]

#then the train test split
from sklearn.model_selection import train_test_split  

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=41)  
#test_size=0.2 means 20% of data frame
#random_state=41 you can take any number insted of 41 it use only for squence output
#train the algo
from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  

regressor.fit(features_train, labels_train) 

print (regressor.predict([[3.073]]))

"""
output==================================================

[-0.59471756]

"""




