
"""
Part 1 Discussions
"""


#https://medium.com/@jayeshbahire/lasso-ridge-and-elastic-net-regularization-4807897cb722

"""
L1 Regularization aka Lasso Regularization– This add regularization terms in the model which are function of absolute value of the coefficients of parameters. The coefficient of the paratmeters can be driven to zero as well during the regularization process. Hence this technique can be used for feature selection and generating more parsimonious model
L2 Regularization aka Ridge Regularization — This add regularization terms in the model which are function of square of coefficients of parameters. Coefficient of parameters can approach to zero but never become zero and hence
Combination of the above two such as Elastic Nets– This add regularization terms in the model which are combination of both L1 and L2 regularization.
"""

#Import packages

import numpy as np 
import pandas as pd 
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#Import dataset and create a dataframe

from sklearn import datasets 
boston = datasets.load_boston ()

#Find out more about this dataset
print(boston.DESCR)

#Boston dataset features
boston.keys() 
boston.data  # ( features )
boston.data.shape
boston.feature_names
boston.target  # label

#Create a dataframe

boston_df	= pd.DataFrame (boston.data, columns= boston.feature_names )
boston_df.head()


#Add dependent Variable
boston_df['House_Price']= boston.target
boston_df.head ()
boston_df.describe()

features = boston_df.drop('House_Price',axis = 1)
labels = boston_df['House_Price']
features.head()
labels.head()

#Create train and test data with 70o/o and 30°/o split
features_train, features_test, labels_train,labels_test	=	train_test_split(features, labels, test_size=0.3, random_state=1)

features_train.shape

features_test.shape

labels_train.shape
labels_test.shape



#Let's import the Lasso, Ridge, Elasticnet regression object and define model 
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge  # RidgeClassier is also there
from sklearn.linear_model import ElasticNet
lm = LinearRegression ()
lm_lasso = Lasso() 
lm_ridge =  Ridge() 
lm_elastic = ElasticNet() 


#Fit a model on the train data
lm.fit(features_train, labels_train)
lm_lasso.fit(features_train, labels_train)
lm_ridge.fit(features_train, labels_train)
lm_elastic.fit(features_train, labels_train)


#Evaluate the model
plt.figure (figsize= (15,10))
ft_importances_lm = pd.Series(lm.coef_, index= features.columns)
ft_importances_lm .plot(kind = 'barh')
plt.show()


#R2 Value

print ("RSquare Value for Simple Regresssion TEST data is-") 
print (np.round (lm .score(features_test,labels_test)*100,2))

print ("RSquare Value for Lasso Regresssion TEST data is-")
print (np.round (lm_lasso.score(features_test,labels_test)*100,2))

print ("RSquare Value for Ridge Regresssion TEST data is-")
print (np.round (lm_ridge.score(features_test,labels_test)*100,2))

print ("RSquare Value for Elastic Net Regresssion TEST data is-")
print (np.round (lm_elastic.score(features_test,labels_test)*100,2))

#Predict on test and training data

predict_test_lm =	lm.predict(features_test ) 
predict_test_lasso = lm_lasso.predict (features_test) 
predict_test_ridge = lm_ridge.predict (features_test)
predict_test_elastic = lm_elastic.predict(features_test)

#Print the Loss Funtion - MSE & MAE

import numpy as np
from sklearn import metrics
print ("Simple Regression Mean Square Error (MSE) for TEST data is") 
print (np.round (metrics .mean_squared_error(labels_test, predict_test_lm),2) )

print ("Lasso Regression Mean Square Error (MSE) for TEST data is") 
print (np.round (metrics .mean_squared_error(labels_test, predict_test_lasso),2))

print ("Ridge Regression Mean Square Error (MSE) for TEST data is") 
print (np.round (metrics .mean_squared_error(labels_test, predict_test_ridge),2))

print ("ElasticNet Mean Square Error (MSE) for TEST data is")
print (np.round (metrics .mean_squared_error(labels_test, predict_test_elastic),2))



"""
The benchmark of random guessing should get you an RMSE = standard_deviation. 
So lower than this, your model is demonstrating some ability to learn; 
above that number, you haven't even learned to guess the mean correctly.
"""







"""
Part 2 Discussions
"""

"""
Ridge Regression and Lasso
This notebook explores ridge regression and lasso. 
These alternative linear fitting techniques can improve a model's performance 
and interpretability.
"""

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Analysis
data = pd.read_csv("data/Advertising.csv")
data.head()

#Drop the first column

data.drop(['Unnamed: 0'], axis=1, inplace=True)

print (data.head())

print (data.columns)

#lets plot few visuals
def scatter_plot(feature, target):
    plt.scatter(data[feature], data[target], c='black')
    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()

scatter_plot('TV', 'sales')
scatter_plot('radio', 'sales')
scatter_plot('newspaper', 'sales')

#Lets build the models now
#Multiple Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

features = data.drop(['sales'], axis=1) #drop the target to get the features
labels = data['sales'].values.reshape(-1,1) #choose the target

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, features, labels, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print(mean_MSE)


#Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

#alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(features, labels)

ridge_regressor.best_params_
ridge_regressor.best_score_


#Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()
"""
For ridge regression, we introduce GridSearchCV. 
This will allow us to automatically perform 5-fold cross-validation with a range of different regularization parameters in order to find the optimal value of alpha.
"""

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(features, labels)

lasso_regressor.best_params_
lasso_regressor.best_score_


#Version 2

# Standardize x_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(features)
features = scaler.transform(features)

from sklearn.linear_model import Ridge, Lasso

# Fit x_train and y_train
ridge = Ridge().fit(features, labels)
lasso = Lasso().fit(features, labels)


# Print results for both Ridge and Lasso
print('Ridge: ', ridge.coef_)
print('Lasso:', lasso.coef_)





"""

Code Challenges 02: (House Data) kc_house_data.csv

This is kings house society data.
In particular, we will: 
• Use Linear Regression and see the results
• Use Lasso (L1) and see the resuls
• Use Ridge and see the score

"""








# skip from here onwards


# coding: utf-8

# This lab on Ridge Regression and the Lasso is a Python adaptation of p. 251-255 of "Introduction to Statistical Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. Adapted by R. Jordan Crouser at Smith College for SDS293: Machine Learning (Spring 2016).
# 
# # 6.6: Ridge Regression and the Lasso



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error


# We will use the `sklearn` package in order to perform ridge regression and
# the lasso. The main functions in this package that we care about are `Ridge()`, which can be used
# to fit ridge regression models, and `Lasso()` which will fit lasso models. They also have cross-validated counterparts: `RidgeCV()` and `LassoCV()`. We'll use these a bit later.
# 
# Before proceeding, let's first ensure that the missing values have
# been removed from the data, as described in the previous lab.



df = pd.read_csv('Hitters.csv').dropna().drop('Player', axis = 1)
df.info()
dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])


# We will now perform ridge regression and the lasso in order to predict `Salary` on
# the `Hitters` data. Let's set up our data:


y = df.Salary

# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')

# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)

X.info()


# Ridge Regression
# The `Ridge()` function has an alpha argument ($\lambda$, but with a different name!) that is used to tune the model. We'll generate an array of alpha values ranging from very big to very small, essentially
# covering the full range of scenarios from the null model containing
# only the intercept, to the least squares fit:


alphas = 10**np.linspace(10,-2,100)*0.5


# Associated with each alpha value is a vector of ridge regression coefficients, which we'll
# store in a matrix `coefs`. In this case, it is a $19 \times 100$
# matrix, with 19 rows (one for each predictor) and 100
# columns (one for each value of alpha). Remember that we'll want to standardize the
# variables so that they are on the same scale. To do this, we can use the
# `normalize = True` parameter:



ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)


# We expect the coefficient estimates to be much smaller, in terms of $l_2$ norm,
# when a large value of alpha is used, as compared to when a small value of alpha is
# used. Let's plot and find out:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# We now split the samples into a training set and a test set in order
# to estimate the test error of ridge regression and the lasso:


# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


# Next we fit a ridge regression model on the training set, and evaluate
# its MSE on the test set, using $\lambda = 4$:


ridge2 = Ridge(alpha = 4, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred2))          # Calculate the test MSE


# The test MSE when alpha = 4 is 106216. Now let's see what happens if we use a huge value of alpha, say $10^{10}$:


ridge3 = Ridge(alpha = 10**10, normalize = True)
ridge3.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred3 = ridge3.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge3.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred3))          # Calculate the test MSE


# This big penalty shrinks the coefficients to a very large degree, essentially reducing to a model containing just the intercept. This over-shrinking makes the model more biased, resulting in a higher MSE.

# Okay, so fitting a ridge regression model with alpha = 4 leads to a much lower test
# MSE than fitting a model with just an intercept. We now check whether
# there is any benefit to performing ridge regression with alpha = 4 instead of
# just performing least squares regression. Recall that least squares is simply
# ridge regression with alpha = 0.



ridge2 = Ridge(alpha = 0, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred = ridge2.predict(X_test)            # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred))           # Calculate the test MSE


# It looks like we are indeed improving over regular least-squares!
# 
# Instead of arbitrarily choosing alpha $ = 4$, it would be better to
# use cross-validation to choose the tuning parameter alpha. We can do this using
# the cross-validated ridge regression function, `RidgeCV()`. By default, the function
# performs generalized cross-validation (an efficient form of LOOCV), though this can be changed using the
# argument `cv`.



ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_


# Therefore, we see that the value of alpha that results in the smallest cross-validation
# error is 0.57. What is the test MSE associated with this value of
# alpha?


ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge4.fit(X_train, y_train)
mean_squared_error(y_test, ridge4.predict(X_test))


# This represents a further improvement over the test MSE that we got using
# alpha $ = 4$. Finally, we refit our ridge regression model on the full data set,
# using the value of alpha chosen by cross-validation, and examine the coefficient
# estimates.

#This is the final model with alpha = 0.57
# L2 Regularization

ridge4.fit(X, y)
pd.Series(ridge4.coef_, index = X.columns)


# As expected, none of the coefficients are exactly zero - ridge regression does not
# perform variable selection!
 

# # The Lasso
# We saw that ridge regression with a wise choice of alpha can outperform least
# squares as well as the null model on the Hitters data set. We now ask
# whether the lasso can yield either a more accurate or a more interpretable
# model than ridge regression. In order to fit a lasso model, we'll
# use the `Lasso()` function; however, this time we'll need to include the argument `max_iter = 10000`.
# Other than that change, we proceed just as we did in fitting a ridge model:


lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# Notice that in the coefficient plot that depending on the choice of tuning
# parameter, some of the coefficients are exactly equal to zero. We now
# perform 10-fold cross-validation to choose the best alpha, refit the model, and compute the associated test error:


lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

#As we get the final value of alpha from above (LassoCV), we make the final model
# L1 Regularization

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
mean_squared_error(y_test, lasso.predict(X_test))


# This is substantially lower than the test set MSE of the null model and of
# least squares, and only a little worse than the test MSE of ridge regression with alpha
# chosen by cross-validation.
# 
# However, the lasso has a substantial advantage over ridge regression in
# that the resulting coefficient estimates are sparse. Here we see that 13 of
# the 19 coefficient estimates are exactly zero:


# Some of the coefficients are now reduced to exactly zero.
pd.Series(lasso.coef_, index=X.columns)


