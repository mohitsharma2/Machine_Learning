# Open Salary_Classification.csv
# Teach how to read the dataset 
# Sandeep is working in the Development Department, 
# have worked for 1000 hours
# Has 1 Certificate and have 2 years experience
# Now predict the salary for Sandeep ??
# Which are the features and labels ??
# So there are 4 features and 1 Label ( Multivariate Data)




# Multiple Linear Regression



# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/Salary_Classification.csv')

# Check data Types for each columns
print(dataset.dtypes)


# NDArray 
# Introduce the concept of Categorical and Numerical Data Types
temp = dataset.values   # dataset.values.tolist()
type(temp)


# Seperate Features and Labels
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

# Check Column wise is any data is missing or NaN
dataset.isnull().any(axis=0)

# Check data Types for each columns
print(dataset.dtypes)

# For applying Regression Algorithm, it is mandatory that there should not be
# any Categorial Data in the dataset
# Decission Tree and Random Forest Algo can take Categorical Data

# Converting your categorical data to Numerical Data is known as Label Encoding


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

# Create objct of LabelENcoder
labelencoder = LabelEncoder()

# Fit and Use the operation Transform
features[:, 0] = labelencoder.fit_transform(features[:, 0])

print(features)
# Development = 0
# Testing = 1
# UX = 2
# There is a problem in Label Encoding, our algo would understand 
# that UX has more preference/priority than Testing and Development
# or reverse 0 is high priority
# So there is a ORDER Generation after Label Encoding 
# So there should be a solution to fix it



# Show One_hot_encoding_colors.png
# Calculate the unique values in Column
# Create new columns for each unique value
# Is your value is equal to Column Name, make it 1 .. others 0
 
# Show One_hot_encoding_week_days.png

# This way of representation of data is known as One Hot Encoding



from sklearn.preprocessing import OneHotEncoder

# Creation of Object
onehotencoder = OneHotEncoder(categorical_features = [0])

# Convert to NDArray format
features = onehotencoder.fit_transform(features).toarray()
# OneHotEncoder always puts the encoded values as the first columns
# irrespective of the column you are encoding

print(features)

# Development = 1 0 0
# Testing = 0 1 0
# UX = 0 0 1


# Now sklearn has new way of using Transformer
# Pandas has pd.get_dummies(dataset) will directly OneHotEncode your data
# Pandas always adds the encoded columns as the last columns

# If there is a new column of Qualification which has B.Tech, M.Tech, Ph.D values
# Then we DO NOT need to OneHotEncode the column, only Label Encoding is required
# Since we do not require to remove the ORDERING issue



"""
#https://towardsdatascience.com/columntransformer-in-scikit-for-labelencoding-and-onehotencoding-in-machine-learning-c6255952731b
#from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
features = np.array(columnTransformer.fit_transform(features), dtype = np.float32)
#x = preprocess.fit_transform(features)

features = features[:,1:]
"""


# If we know the values of first two columns, then can we know the value in third col
# That means one column out of three is redundant
# And its value is dependent on the other columns
# Predictions will be poor if there are redundant columns in dataset
# This problem is known as Dummy Variable Trap
# We can drop any one column to solve this problem 
# We should have singularity in the dataset and not have redundancy

 

# Avoiding the Dummy Variable Trap
# dropping first column
features = features[:, 1:]
# Now our data is ready for Modelling 



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
# Whether we have Univariate or Multivariate, class is LinearRegression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)


# Check the value of intercept and slope
# y = ax + by + cz + d
# Here a, b and c are the coefficients and d is the intercept
print(regressor.intercept_)   
print (regressor.coef_)   # we will have 5 values since we have 5 columns (5dimension)
# We cannot show a line on a graph as we did for 2D data, since we have 5D data



# Predicting the Test set results
Pred = regressor.predict(features_test)

import pandas as pd
import numpy as np
print (pd.DataFrame(zip(Pred, labels_test)))


# Prediction for a new values for a person in 'Development', hours worked 1150,
# 3 certificates , 4yrs of experience. What would be his salary ??


regressor.predict(['Development',1150,3,4])
# This will throw error of Expected 2D Array but got 1D Array 



x = ['Development',1150,3,4]

print(type(x))
x = np.array(x)
print(type(x))

print(x.shape)
x = x.reshape(1,4) # x.reshape(1,-1)
print(x.shape)


regressor.predict(x)
# This again throws error, cannot cast array data from float to x32
# Since it requires 5 columns but you are passing only 4 columns

 
# make this according to the data csv format
# We need to OneHotEncode the data
# Development is replaced by 1,0,0 to 0,0 to remove dummy trap

x = [1,0,0,1150,3,4]
x = np.array(x)
x = x.reshape(1,4)
regressor.predict(x)
# Again we will get error



# We need to remove the dummy trap also
# Development 1,0,0 to be replaced by 0,0 to remove dummy trap

x = [0,0,1150,3,4]
x = np.array(x)
x = x.reshape(1,5)
regressor.predict(x)


# General Way of solving the above problem
le = labelencoder.transform(['Development'])
print(le)
# le = labelencoder.transform(['Testing'])
# print(le)
# le = labelencoder.transform(['UX'])
# print(le)
# le = labelencoder.transform(['CSE'])
# print(le)

ohe = onehotencoder.transform(le.reshape(1,1)).toarray()
print(ohe)
x = [ohe[0][1],ohe[0][2],1150,3,4]


x = np.array(x)
x = x.reshape(1, -1)
regressor.predict(x)



"""
x = np.array(columnTransformer.transform(x), dtype = np.float32)
x = x[:,1:]
regressor.predict(x)
"""


# Getting Score for the Multi Linear Reg model
Score = regressor.score(features_train, labels_train)
Score = regressor.score(features_test, labels_test)


"""
If the training score is POOR and test score is POOR then its underfitting
If the training score is GOOD and test score is POOR then its overfitting
"""



"""
Steps in Data Preprocessing

Step 1 : Import the libraries
Step 2 : Import the data-set
Step 3 : Check out the missing values - imputation using sklearn, pandas
Step 4 : Label encoding - categorical data using LabelEncoder, cat.code (category)
Step 5 : order issue - onehotencoding (dummy encoding) - OneHotEncoder, get_dummies
Step 6 : Splitting the data-set into Training and Test Set
Step 7 : Feature Scaling

"""



# Code Challenge


"""
HandsOn
#Import the local file cars.csv and split the data set equally into test set and training set.
#Print it and save both data sets  into two new .csv file.

"""



import pandas as pd
import numpy as np


# Importing the dataset
dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/cars.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1,1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
print (X_train,X_test,y_train, y_test)


# Write code to save in the csv file

# Combining the features and labels in both train and test data
train_data = np.concatenate([X_train, y_train],axis=1)
test_data = np.concatenate([X_test, y_test], axis=1)

# Fetching all the columns name from the original dataset
head = list(dataset.columns)

# Framing the test and train dataframe
train_df, test_df = pd.DataFrame(), pd.DataFrame()

for var in range(0,12):
    train_df[head[var]] = train_data[:, var]
    test_df[head[var]] = test_data[:, var]

# Creating seperate train and test csv files
train_df.to_csv("data/cars_train.csv")
test_df.to_csv("data/cars_test.csv")

# Printing the train and test dataframes
print("train_data:", train_df)
print("test_data:", test_df)


"""

Q1. (Create a program that fulfills the following specification.)
iq_size.csv

Are a person's brain size and body size (Height and weight) 
predictive of his or her intelligence?

 

Import the iq_size.csv file

It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students

Column 2:  The brain size (MRI) of students (given as count/10,000).

Column 3: The height (Height) of students (inches)

Column 4: The weight (Weight) of student (pounds)

    What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
    Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or brain size.




Q2. (Create a program that fulfills the following specification.)
Female_Stats.Csv

Female Stat Students

 

Import The Female_Stats.Csv File

The Data Are From N = 214 Females In Statistics Classes At The University Of California At Davi.

Column1 = Student’s Self-Reported Height,

Column2 = Student’s Guess At Her Mother’s Height, And

Column 3 = Student’s Guess At Her Father’s Height. All Heights Are In Inches.

 

Build A Predictive Model And Conclude If Both Predictors (Independent Variables) Are Significant For A Students’ Height Or Not
When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.

"""






