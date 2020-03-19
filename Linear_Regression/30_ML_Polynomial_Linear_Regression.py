# Simple Linear Regression ( Straight Line )
# y = mx + c 

# Multiple Linear Regression ( Straight Line )
# y = ax + by + cz + d

# Polynomial Regression ( Polynomial Curve )
# y = ax² + bx + c ( Quadratic Equation )
# Feature is one only, but we have converted into different degrees, 
# multiplied by different coefficients

# Open Claims_Paid.csv
# Explain the Insurance Domain and How claims are paid
# Expalin the columns
# How much they are going to Pay in 1981 
# It seems it is a ML Problem (LR) and as the years are inc cost is inc



# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/Claims_Paid.csv')
dataset.ndim
dataset.shape

# Check for categorical data
dataset.dtypes 

# Check for missing data
dataset.isnull().any(axis=0)


# Seperate features and labels
features = dataset.iloc[:, 0:1].values
print(features)

labels = dataset.iloc[:, 1].values
print(labels)


#let's first analyze the data
# Visualising the Linear Regression results
# There seems to be a positive corelation between feature and labels

plt.scatter(features, labels)


# Fitting Linear Regression to the dataset
# We can avoid splitting since the dataset is too small

from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(features, labels)



print ("Predicting result with Linear Regression")
print (lin_reg_1.predict([[1981]])) # if error we can pass as a list


# Visualising the Linear Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, lin_reg_1.predict(features), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Year')
plt.ylabel('Claims Paid')
plt.show()

# After seeing the visual, its sems that the predictions will be poor
# Once the years increases
# Actual line should be a polynomial line
# What should be the degree of the polynomial function 
# Its a hit and trail method and visualize it to see the curve


# We need to convert the feature(x) into 5 degree format
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_object = PolynomialFeatures(degree = 5)
print(features.shape)

features_poly = poly_object.fit_transform(features)
print(features_poly)
print(features_poly.shape) # x0 x1 x2 x3 x4 x5    


# Algo is same for Polynomial Regression, its only the data format is changed
lin_reg_2 = LinearRegression()
lin_reg_2.fit(features_poly, labels)


# This will give error dim1 != dim 6
# We need to convert the data into polynomial format
print (lin_reg_2.predict(1981))



print ("Predicting result with Polynomial Regression")
print (lin_reg_2.predict(poly_object.transform(1981)))
# This value has huge difference from the Linear Regression 
# But if we visualize it, we will come to know that Poly is better predictions



# Visualising the Polynomial Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, lin_reg_2.predict(poly_object.fit_transform(features)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Year')
plt.ylabel('Claims Paid')
plt.show()

# Try with different degrees = 2 and then visualize it

# Try with different degrees = 8 and then visualize it


# What if the data is Multivariate, 
# we can then also apply Polynomial Regression
# Since in the above example we have taken Univariate dataset


# Open Salary_Classification.csv
# We used this for Multiple Linear Regression

"""
    +++++++++++++++++++++++
    |Department        |0 |
    |WorkedHours       |1 |
    |Certification     |2 |
    |YearsExperience   |3 |
    |Salary            |4 | 
    +++++++++++++++++++++++
"""

# But which out of 4 features actually affect the most in predicting the salary ?
# Multiple features can be also important in prediction 

# Another analogy would be the Upper Managment to HR for lot of resignation


# There is a Solution to solve the above problem 
# There are two approaches
# 1. Forward Elimination 
# 2. Backward Elimination 
    

# Importing the libraries
import pandas as pd
import numpy as np


# Importing the dataset
dataset = pd.read_csv('data/Salary_Classification.csv')

temp = dataset.values
print(temp)

features = dataset.iloc[:, :-1].values
print(features)

labels = dataset.iloc[:, -1].values
print(labels)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])

# One Hot Encoding 
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()
print(features)

# Avoiding the Dummy Variable Trap
# dropping first column
features = features[:, 1:]
print(features)


# We are not performing Prediction today
# But we have to identify which is the most important columns
# For which we will use a new library of statsmodels and not sklearn
# This library also has methods to perform linear Regression 
# OLS ( Optimal Least Square ) is the method to perform LR
# Refer to the least square regression line image or concept



# Building the optimal model using Backward Elimination
import statsmodels.api as sm


'''
output = ax + by + cz + d
a,b,c are known as weights / coefficients
a is the slope on x axis
b is the slope on y axis
c is the slope on z axis
d is the intercept (d feature is multiplied by constant 1 )

output = d(1) + ax + by + cz
d(1) is taken care by LinearRegression Class, but not by OLS
We need to add in the OLS from outside
'''

#This is done because statsmodels library requires it to be done for constants.
#features = np.append(arr = np.ones((30, 1)), values = features, axis = 1)

#adds a constant column to input data set.
features = sm.add_constant(features)
print(features)

"""
Unlike SKLearn, statsmodels doesn’t automatically 
fit a constant, so you need to use the method 
sm.add_constant(features) in order to add a constant. 
Adding a constant, while not necessary, 
makes your line fit much better. 
For example, if you have a line with an intercept of 
-2000 and you try to fit the same line through the 
origin, you’re going to get an inferior line. 
Once we add a constant (or an intercept if you’re 
thinking in line terms), you’ll see that the 
coefficients are the same in SKLearn and statsmodels.
"""




# Assume all features are important, after analysis we will drop some 
# features which are not important
# We have created a copy of the features
'''
    ++++++++++++++++++++++++++
    |constant          |0 |  |
    |Dept Dummy 1      |1 |x1|
    |Dept Dummy 2      |2 |x2|
    |WorkedHours       |3 |x3|
    |Certification     |4 |x4|
    |YearsExperience   |5 |x5|
    ++++++++++++++++++++++++++
'''
features_opt = features[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit() # Applying LR
# exog = features
# endog = labels
# fit will train the model
# We need to focus on the P value of the Summary
regressor_OLS.summary()
#regressor_OLS.pvalues


'''
Visualise the Summary for p value p > |t|
Column whose p value is highest is the least imporant feature
p > 5% should be removed in the rule
We can drop that column in the next iteration
x2 is having the highest p value, so remove it, since it is least important
x2 = Dept Dummy Variable 2
    ++++++++++++++++++++++++++
    |constant          |0 |c |
    |Dept Dummy 1      |1 |x1|
    |Dept Dummy 2      |2 |x2|
    |WorkedHours       |3 |x3|
    |Certification     |4 |x4|
    |YearsExperience   |5 |x5|
    ++++++++++++++++++++++++++
x2 needs to be removed, that is at index 2 from original features 
'''

features_opt = features[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
#regressor_OLS.pvalues


'''
    ++++++++++++++++++++++++++
    |constant          |0 |c |
    |Dept Dummy 1      |1 |x1|
    |Dept Dummy 2      |2 |  |x
    |WorkedHours       |3 |x2|
    |Certification     |4 |x3|
    |YearsExperience   |5 |x4|
    ++++++++++++++++++++++++++

'''
features_opt = features[:, [0, 1, 3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()



'''
    +++++++++++++++++++++++++++++
    |constant          |0 |c |c |
    |Dept Dummy 1      |1 |x1|x1|
    |Dept Dummy 2      |2 |  |  |x
    |WorkedHours       |3 |x2|x2|
    |Certification     |4 |x3|  |x
    |YearsExperience   |5 |x4|x3|
    +++++++++++++++++++++++++++++

'''
features_opt = features[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()


'''
    ++++++++++++++++++++++++++++++++
    |constant          |0 |c |c |c |
    |Dept Dummy 1      |1 |x1|x1|  |x
    |Dept Dummy 2      |2 |  |  |  |x
    |WorkedHours       |3 |x2|x2|x1|
    |Certification     |4 |x3|  |  |x
    |YearsExperience   |5 |x4|x3|x2|
    ++++++++++++++++++++++++++++++++

'''
features_opt = features[:, [0, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
print (regressor_OLS.summary())

'''
    +++++++++++++++++++++++++++++++++++
    |constant          |0 |c |c |c |c |
    |Dept Dummy 1      |1 |x1|x1|  |  |x
    |Dept Dummy 2      |2 |  |  |  |  |x
    |WorkedHours       |3 |x2|x2|x1|  |x
    |Certification     |4 |x3|  |  |  |x
    |YearsExperience   |5 |x4|x3|x2|x1|
    +++++++++++++++++++++++++++++++++++
    
    Nothing to remove now
    Both const and x1 (Years Experience) is needed, since the p value is < 5%
    Experience is the most important feature to control the salary label
'''


# Define p Value ?
# p value is a  Probability that says your column by chance 
# has come in the decission making process
    


# add code to automate the p value removing
import statsmodels.api as sm
import numpy as np

features_obj = features[:, [0,1,2,3,4]]
features_obj = sm.add_constant(features_obj)
while (True):
    regressor_OLS = sm.OLS(endog = labels,exog =features_obj).fit()
    p_values = regressor_OLS.pvalues
    if p_values.max() > 0.05 :
        features_obj = np.delete(features_obj, p_values.argmax(),1)
    else:
        break
  
    
    

# Code Challenges
"""

Q. (Create a program that fulfills the following specification.)
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




Q. (Create a program that fulfills the following specification.)
bluegills.csv

How is the length of a bluegill fish related to its age?

In 1981, n = 78 bluegills were randomly sampled from Lake Mary in Minnesota. The researchers (Cook and Weisberg, 1999) measured and recorded the following data (Import bluegills.csv File)

Response variable(Dependent): length (in mm) of the fish

Potential Predictor (Independent Variable): age (in years) of the fish

    How is the length of a bluegill fish best related to its age? (Linear/Quadratic nature?)
    What is the length of a randomly selected five-year-old bluegill fish? Perform polynomial regression on the dataset.

NOTE: Observe that 80.1% of the variation in the length of bluegill fish is reduced by taking into account a quadratic function of the age of the fish.





Q. (Create a program that fulfills the following specification.)
iq_size.csv

Are a person's brain size and body size (Height and weight) predictive of his or her intelligence?

 

Import the iq_size.csv file

It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students

Column 2:  The brain size (MRI) of students (given as count/10,000).

Column 3: The height (Height) of students (inches)

Column 4: The weight (Weight) of student (pounds)

    What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
    Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or brain size.
"""    
    

# Skip this part
"""
Few comments about OLS for dummy variable values

Case Study
Suppose you are building a linear (or logistic) regression 
model. In your independent variables list, you have a 
categorical variable with 4 categories (or levels). 
You created 3 dummy variables (k-1 categories) and 
set one of the category as a reference category. 
Then you run stepwise / backward/ forward regression 
technique and you found only one of the category coming 
out statistically significant based on p-value and the 
remaining 3 categories are insignificant. 
The question arises - should we remove or keep these 3 
categories having insignificant difference? should we 
include the whole categorical variable or not?

Solution
In short, the answer is we can ONLY choose whether we 
should use this independent categorical variable as a 
whole or not. In other words, we should only see whether 
the categorical variable as a whole is significant or not. 
We cannot include some categories of a variable and exclude 
some categories having insignificant difference.

Ref: https://www.listendata.com/2016/07/insignificant-levels-of-categorical-variable.html
"""



"""
In most of the cases library takes care of dummy variable trap and feature scaling as well.
//Explain the sample code

// Compare the prediction with actual data points.

// Now how you can improve the model?
//By checking which independent variables has highest impact?
// For this we use method  called backward elimination
// Explain the backward elimination
//Steps
1. Start with all the predictors in the model
2. Remove the predictor with highest p-value greater than 5%
3. Refit the model and goto 2
4. Stop when all p-values are less than 5%.

// Need to use library for this (import stats_models.formula.api as sm)

//Using this libarary, we remove predictors(independent variables) iteratively by looking at p value (remove if p is more than 5%)

// Most important

https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/


"""



"""
https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
"""
"""
Adding a non linearity into linear
https://www.quantinsti.com/blog/polynomial-regression-adding-non-linearity-to-a-linear-model/
"""




