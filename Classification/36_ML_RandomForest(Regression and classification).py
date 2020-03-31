"""
RandomForest

There is a Mela in Sandeep town
There is a Bull there, and now people are asked to predict its weight

This is a Regression Problem, since we have to predict a weight which is a
continuous value

It might happen that the person who is a rural will give near prediction
But a person from urban would give not so near prediction
But both are not correct

But if we take the average of the prediction, then it was very close

We use this same approach for RandomForest 
    
Lets assume a dataset has 10,000 observations
Randomly pick up a sample of 600 observations
Create a Decision Tree for this dataset
    
Now again randomly pick up another sample of 600 observation 
Create a Decision Tree for this dataset again
    
Like this create 25 decision Tree for 25 different 
sample dataset of 600 observations each
    
Average out on this 25 Decision Tree 
This is known as Ensemble Learning 
Ensemble means Average 
    
"""


"""
How the Random Forest Algorithm Works
The following are the basic steps involved in performing the random forest algorithm:

Pick N random records from the dataset.
Build a decision tree based on these N records.
Choose the number of trees you want in your algorithm and repeat steps 1 and 2.
In case of a regression problem, for a new record, each tree in the forest predicts a value for Y (output). 
The final value can be calculated by taking the average of all the values predicted by all the trees in forest. 
Or, in case of a classification problem, each tree in the forest predicts the category to which the new record belongs. 
Finally, the new record is assigned to the category that wins the majority vote.
"""

"""
Advantages of using Random Forest
As with any algorithm, there are advantages and disadvantages to using it.
In the next two sections we'll take a look at the pros and cons of using 
random forest for classification and regression.

The random forest algorithm is not biased, since, there are multiple trees 
and each tree is trained on a subset of data. Basically, the random forest 
algorithm relies on the power of "the crowd"; therefore the overall 
biasedness of the algorithm is reduced.
This algorithm is very stable. Even if a new data point is introduced 
in the dataset the overall algorithm is not affected much since new data 
may impact one tree, but it is very hard for it to impact all the trees.
The random forest algorithm works well when you have both categorical and numerical features.
The random forest algorithm also works well when data has missing values 
or it has not been scaled well (although we have performed feature scaling 
in this article just for the purpose of demonstration).

Disadvantages of using Random Forest
A major disadvantage of random forests lies in their complexity. 
They required much more computational resources, owing to the large number of decision trees joined together.
Due to their complexity, they require much more time to train than other comparable algorithms.
"""


"""
Part 1: Using Random Forest for Classification
"""

"""
Problem Definition
The task here is to predict whether a bank currency note is authentic or not 
based on four attributes i.e. variance of the image wavelet transformed 
image, skewness, entropy, and curtosis of the image.
"""

import pandas as pd

dataset = pd.read_csv("C:/Users/mohit/Desktop/Machine_Learning/Classification/bill_authentication.csv")  

#data analysis
dataset.shape

# Checking for Categorical Data
dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(100)


# Finding missing data
dataset.isnull().any(axis=0)


features = dataset.iloc[:, 0:4].values  
labels = dataset.iloc[:, 4].values 


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.2, random_state=0)  


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test) 

#train the model
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train)  


labels_pred = classifier.predict(features_test) 

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)


# Evaluating score
# For classification tasks some commonly used metrics are confusion matrix, 
# precision, recall, and F1 score.
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(labels_test, labels_pred)
print(cm)  

# Model Score = 98.90 times out of 100 model prediction was RIGHT
print( (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))


#Evaluate the algo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(labels_test,labels_pred))  
print(classification_report(labels_test,labels_pred))  
print(accuracy_score(labels_test, labels_pred))

#To improve the accuracy, I would suggest you to play around with other 
#parameters of the RandomForestClassifier class and see if you can improve on our results.




"""
Part 2: Using Random Forest for Regression
"""

"""
Problem Definition
The problem here is to predict the gas consumption (in millions of gallons) 
in 48 of the US states based on petrol tax (in cents), per capita income 
(dollars), paved highways (in miles) and the proportion of population with 
the driving license.
"""

#Import libraries
import pandas as pd  
import numpy as np  

#import database
dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/Classification/petrol_consumption.csv') 


#data analysis
dataset.shape


# Checking for Categorical Data
dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(10)


# Finding missing data
dataset.isnull().any(axis=0)


#Preparing the data for training
features = dataset.iloc[:, 0:4].values  
labels = dataset.iloc[:, 4].values  


#Training test split
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.2, random_state=0) 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test)  

#train the model
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=25, random_state=0)  
regressor.fit(features_train, labels_train)  

labels_pred = regressor.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
print(df) 


#Evaluating the algorithm
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred)))  
print (np.mean(labels))



"""
With 20 trees, the root mean squared error is 64.93 which is greater than 
10 percent of the average petrol consumption i.e. 576.77. 
This may indicate, among other things, that we have not used enough estimators (trees).
"""

#Change the number of estimators
regressor = RandomForestRegressor(n_estimators=300, random_state=0)  
regressor.fit(features_train, labels_train)  


labels_pred = regressor.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
print(df) 


#Evaluating the algorithm with 300 trees

print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred)))  

# Skip from here onwards







  
# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')
features = dataset.iloc[:, 1:2].values
labels = dataset.iloc[:, 2].values


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(features, labels)

# Predicting a new result
labels_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results (higher resolution)
features_grid = np.arange(min(features), max(features), 0.01)
features_grid = features_grid.reshape(-1, 1)
plt.scatter(features, labels, color = 'red')
plt.plot(features_grid, regressor.predict(features_grid), color = 'blue')
plt.title('(Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


"""
# Decision Treee plotting



from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


"""

"""
Version 2: 
"""

regressor.tree_.children_left #array of left children
regressor.tree_.children_right #array of right children
regressor.tree_.feature #array of nodes splitting feature
regressor.tree_.threshold #array of nodes splitting points
regressor.tree_.value #array of nodes values



from inspect import getmembers
print( getmembers( regressor.tree_ ) )

zip(dataset.columns[regressor.tree_.feature], regressor.tree_.threshold, regressor.tree_.children_left, regressor.tree_.children_right)






""""
Entropy. A decision tree is built top-down from a 
root node and involves partitioning the data into 
subsets that contain instances with similar values 
(homogenous). ID3 algorithm uses entropy to calculate 
the homogeneity of a sample.

http://www.saedsayad.com/decision_tree.htm
"""
"""
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
"""
"""
https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176
https://pythonprogramminglanguage.com/decision-tree-visual-example/
http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/

"""
"""
Parameter tuning:
https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

"""

# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/sd.csv')
features = dataset.iloc[:, 1:2].values
labels = dataset.iloc[:, 2].values


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(features, labels)

# Predicting a new result
labels_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results (higher resolution)
features_grid = np.arange(min(features), max(features), 0.01)
features_grid = features_grid.reshape(-1, 1)
plt.scatter(features, labels, color = 'red')
plt.plot(features_grid, regressor.predict(features_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()