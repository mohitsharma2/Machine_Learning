"""
Lets assume y = f(x)
Where 
    y is the dependent variable
    x is the independent variable
    f is the mapping function 

Lets assume for a set of x and y points
x = [0,1,2,3,4,5]
y = [0,1,2,3,4,5]

For what value of x = 7, y = ?

Mathematically
If we plot the values on a graph paper, it will be a straight line

Show ml-slope-intercept3.png
y = mx + c 

If we can find the slope(m) and intercept of the line (c), 
we can find for x = 7
what will be the value of y 


slope (m) = (y2 - y1) / (x2 - x1)
for any set of value if we calculate it
m = (2 - 1) / ( 2 - 1)
  = 1
  
Since the straight line is passing through the origin, the intercept would be 0
c = 0

So the final equation would be 
y = (1)*x + (0)
y = x

So if x = 7, then y would be also 7

show ml-slope-intercept.jpg

"""

# Open student_scores.csv 
# Identify which is x ( feature or independent ) and y ( label or dependent )
# This is a labelled dataset

# If Sandeep studied for 10 hours then what would be his score

# Solution  """10 hours==features and score is labels"""

# Since we have the x and y values, we can draw a straight line on the graph

# Then we can find the intercept and slope of the line

# after that we can apply  that on the equation y = mx + c and find the value
# of y for a specific value of x





#Importing Libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  


#imports the CSV dataset using pandas
dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/student_scores.csv')


#explore the dataset
print (dataset.shape)
print (dataset.ndim)
print (dataset.head())
print (dataset.describe())

#check data types for each column
print (dataset.dtypes)

#Check if any NaN values in dataset
dataset.isnull().any(axis=0)

# Check for outlier values
# Also helps in finding the data distribution 
plt.boxplot(dataset.values)


# In the last example 
x = [0,1,2,3,4,5]
y = [0,1,2,3,4,5]
plt.axis([0, 6, 0, 6])

# This is how the points are scattered
plt.scatter(x,y)

# Graphically if we see the slope is 1 and incercept is 0
plt.plot(x,y)



# let's plot our data points on 2-D graph to eyeball our dataset 
# and see if we can manually find any relationship between the data. 
dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# But in this case as compared to earlier, its not possible to draw a 
# straight line which passes through all the points

# Take exmaple of Home Decission on a specific topic
# will have different opinion
# As a democratic solution we take such a solution which is acceptable to 
# every one. 
 
# So now we need to draw a straight line, which passes through all the points
# But we would be not be able to satisfy all the points
# To be called a BEST FIT LINE

# Now how to draw the BEST FIT LINE

# Show the ml-leastsquares-regression.jpg
# Explain the images for the three points d1, d2, d3
# This line should minimize the distances from the line
# That would be called as the BEST FIT LINE


# Show ml-LR-Residual.jpg
# Multiple lines can be drawn
# But which line should we draw
# BEST FIT LINE should minimize the sq of vertical distances of each points
# That means we need to find the slope and intercept of the line
# Then we can find the prediction, once the line is drawn
    
# The algorithm for LR would now analyse the data and try to find the 
# BEST FIT LINE 

# Since we have a labelled dataset
# We need to segregate the features and labels

#prepare the data to train the model
features = dataset.iloc[:, :-1]
print(type(features))
print(type(features.values))
                        
features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 1].values 

# Open in the Variable Explorer
type(features)
features
features.shape
features.ndim


# Open in the Variable Explorer
type(labels)
labels
labels.shape
labels.ndim


"""
Now use a LinearRegression algorithm to 
train the model now 
"""

# import the scikit learn library to import the algorithm
from sklearn.linear_model import LinearRegression  

# Create the object of the class, which is known as MODEL
regressor = LinearRegression()  

# Now we need to train the MODEL to make him learn the features and labels
regressor.fit(features, labels) 

# To see the value of the intercept and slop calculated by the linear regression 
# algorithm for our dataset, execute the following code.
print(regressor.intercept_)  # intercept of the line (2.483673405373196)
print (regressor.coef_) # slope of the line (9.77580339)


# what would be prediction of score if someone studies 10 hours
# y = (m)x + (c)   where m is the slope and c is the inetrcept
print (regressor.coef_*10 + regressor.intercept_) # [100.24170731]

"""
#either you can do above last 3 steps or you can do below 1 step for prediction
"""

# This can also be predicted using a function 
print (regressor.predict([[10]])) # [100.24170731]  this is a label
##or
#labels=regressor.predict([[10]])
#print(labels)

#Visualize the best fit line
import matplotlib.pyplot as plt


# Visualising the  results
plt.scatter(features, labels, color = 'red')
plt.plot(features, regressor.predict(features), color = 'blue')
plt.title('Study Hours and Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score: Marks')
plt.show()


# Explain for Actual data 2.5 = 21 Vs the predicted values graphically
# Graphically it will be around 27
# Difference of Actual y and predicted y should be minimum for the BEST FIT LINE
# The point which is farthest from the BEST FIT LINE would have the worst prediction 




# Now if I want to get 100 marks
# How much hours do i need to study 

# Reverse the features and labels and try the model and prediction again
# You might need to reshape the features by features = features.reshape(25,1)
# Now regressor.predict(100)


# Since you have only 1 feature, so this is also called Simple Linear Regression
# or Univariate 
# if the features are more than 1, then it is known as multivariate or Multiple
# Linear Regression


 
# A faculty in your college teaches you Data Structure
# How does the faculty EVALUATES that how much have you understood that 
# he has explained or TRAINED you
# How well your Brain is trained WELL or BAD is EVALUATED
# TEST or EXAM is the way to EVALUATE it

# There is a Data Structure Book(DATASET), through which your BRAIN needs to be 
# TRAINED

# There are 100 QUESTIONS and Corresponding 100 ANSWERS
# Questions can be considered as FEATURES and Answers as LABELS
# That means its an Labelled Dataset
# So during the classes in college, he would train you on 60Q and 60A

# Now he would take TEST and ask you some questions and you need to answer them
# and then we would compare your answers with his set of answers 

# Lets assume now he gives you 40Q
# Lets assume you have given 40Q' == 40A

# The better the comparison score, the better is the Training of your Brain


# We have features and labels 
# Lets split into two parts
# features_train and labels_train
# features_test and labels_test 

# Now lets train our model 
# .fit(features_train, labels_train)
# Now predict 
# pred = .predict(features_test)

# Now we can compare the pred == labels_test

# As a thumb Rule we would use 60-80% for training and rest for testing 



#Importing Libraries
import pandas as pd  
import matplotlib.pyplot as plt  


#imports the CSV dataset using pandas
dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/student_scores.csv')  


#prepare the data to train the model
features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 1].values 


#and then the train test split
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=41)  

# If we combine features_train, features_test then we will get features
# If we combine labels_train, labels_test then we will get labels


#train the algo
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features_train, labels_train) 


# There would be change in the intercept and slope since the data is less
print(regressor.intercept_)  # intercept of the line (1.0541144761913515)
print (regressor.coef_) # slope of the line (9.77580339)


"""
This means that for every one unit of change in hours studied, 
the change in the score is about 9.91%. 
"""

# making predictions
# To make pre-dictions on the test data, execute the following script:

labels_pred = regressor.predict(features_test) 

#To compare the actual output values for features_test with the predicted values, execute the following script 
df = pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred})  
print (df) 


# Show ml-leastsquares-regression.jpg
# So its a mathematical problem we need to solve
# Ultimately we need to minimize the d1, d2, d3 
# We need to identify the intercept and slope 
# so that the equation is minimized
# So its an mathematics optimisation problem 

# That algo which helps in finding the exact value to minimize it is known as GD
# GD = Gradient Descent
# There are different approaches for GD to minimize it 
# Like Newtons Method
# In Machine Learning the method  is known as Gradient Descent
# This expression is of sqr(d1) + sqr(d2) + sqr(d3) ...  = Cost/Loss Function 


# So the target of the Algo/Model is to minimize the Cost/Loss Function 
# the method of minimizing the Cost / Loss function is known as GD

    

#Now explain the random state by pring features_test with and without random state set

import numpy as np

# Simultaneous Assignment
X, y = np.arange(10).reshape((5, 2)), list(range(5))

print (type(X))
print (type(y))

print (X)
print (y)


# train_test_split splits arrays or matrices into random train and test subsets. 
# That means that everytime you run it without specifying random_state, 
# you will get a different result, this is expected behavior. 



#from sklearn.cross_validation import train_test_split
#Deprecated since version 0.18: This module will be removed in 0.20. 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print (X_train)
print (X_test)

print (y_train)
print (y_test)



# If you use random_state=some_number, 
# then you can guarantee that your split will be always the same. 
# This is useful if you want reproducible results, 
# I would say, set the random_state to some fixed number while you test stuff, 
# but then remove it in production if you need a random (and not a fixed) split.



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print (X_train)
print (X_test)

print (y_train)
print (y_test)


# This result would be different from last one, but if you run it again and again it will be same

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

print (X_train)
print (X_test)

print (y_train)
print (y_test)



# Model accuracy
print (regressor.score(features_test, labels_test))
print (regressor.score(features_train, labels_train))



# Now lets talk about two terms - overfitting and underfitting the data


# Do we have case of underfitting?
# Do we have case of overfitting?


"""
If the training score is POOR and test score is POOR then its underfitting ( no padai ) 

If the training score is GOOD and test score is POOR then its overfitting ( ratoo tota )
"""


# Explain the logic of Best Teacher 100 question story


"""
# Underfitting = no padai

It means that the model does not fit the training data and therefore misses 
the trends in the data
this is usually the result of a very simple model (not enough predictors/independent variables).
"""
 


"""
# Overfitting = ratoo tota

This model will be very accurate on the training data but will probably be very 
not accurate on untrained or new data

This usually happens when the model is too complex (i.e. too many features/variables 
compared to the number of observations). 

It is because this model is not generalized 
Basically, when this happens, the model learns or describes the “noise” in the 
training data instead of the actual relationships between variables in the data.

"""


"""
Solution to Underfitting
    Increase Training Data
    Change the Model from simpler to Complex 
    

Solution to Overfitting
 There are two types of regularization as follows:

    L1 Regularization or Lasso Regularization
    L2 Regularization or Ridge Regularization
    Elastic Net is hybrid of both L1 and L2
"""



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


"""

Explain the Income Data fields

Divide into features and labels with explanation 

HR Tool to offer a salary to new candidate of 15 yrs

Explain the plotting of two list with range data

Explain best fit line concept, Calculate slope and constant  
y = mx + c 

Gradient Descent concept to find best fit line
to minimise the loss or cost function

For Linear Regression the cost function = Mean Square error
its diffetent for different algorithms 



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Income_Data.csv')

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

"""
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split (features, labels, test_size = 0.2, random_state = 0)

"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(features, labels)

x = [12]

x = np.array(x)
x = x.reshape(1,1)
regressor.predict(x)


# Now with test and train data 

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2,
                random_state = 0)

regressor = LinearRegression()
regressor.fit(features_train, labels_train)


# Predicting the Test set results
labels_pred = regressor.predict(features_test)


print (pd.DataFrame(labels_pred, np.round(labels_test,2)))




# Visualising the Training set results
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Training set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()


# Visualising the Test set results
#plt.scatter(features_train, labels_train, color = 'green')
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Test set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()


#Model accuracy
print (regressor.score(features_test, labels_test))
print (regressor.score(features_train, labels_train))
"""



