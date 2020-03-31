# Decission Tree for Classification

"""
Decission Tree and Random Forest are also known as CART Algorithm

Tree Data Structure in C
Binary Search Tree in C
You organise your data in a form of Tree so that your search becomes fast
You have a Root Node and then you take a decision whether you go to LEFT Node
or RIGHT Node, till we reach the Leaf Node

In Decisson Tree Algorithm, we use our training data to create a Tree 
and then for the new points we use this tree structure to predict the postion 
of the new points

https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/

There are 14 instances of golf playing decisions based on 
outlook, temperature, humidity and wind factors

Explain each features 

Golf Decission Dataset
features = Outlook, Temp, Humidity, Wind
label    = Decision

So is this a Classification Problem or a Regression Problem ?
There are 2 Classes ( YES or NO )   

Make a decission Tree based on the data.

open cart-Decission-Tree.png

Now predict whether we should play Golf on 
Sunny(outlook), Normal(Humidity), Mild(Temp), Weak (Wind) 

Now Traverse the Tree based on the predicted data
You will traverse any path, but eventually reach the decission ( YES or NO )

Now when the Decisson Tree in the leaf Node does not have a class, but have a
continuous value, then it would be a Regression.

Now see how to create this Tree 




There are two algorithms to create decision tree ( ID3 and CART )      

ID3  ----> Entropy based and your calculate  Information Gain
These are statistics keywords
Entropy is the measurement of Randomness 

Today we will use CART algo for making the decission tree
CART  ----> Gini Index/Score


How to define Gini Index / Score ?
Gini Index is the measurement of impurity 
More impure, more the Gini Index

What do you mean by Impure ?
If by looking at the feature, if I am NOT ABLE to take decission, 
then it is IMPURE, Ginni index will be MORE

If by looking at the feature, if I am ABLE to take decission, 
then it is PURE, Ginni index will be LESS 

How to calculate the Gini Index ?
Gini index would be calculated based on the unique values in it

Now with the above knowlege, we would iterate through the data and create
the Tree

Who would be the root node / Nodes 
Find the gini index of all the 4 features, which has the lowest gini index 
would be the Root Node and similarly we need to iterate the same logic

    
How to find Gini Index for Outlook feature ?
Dry run the code on the web page for all the features


Time to decide ?
--------------------------
Feature     | Gini index
--------------------------
Outlook   	| 0.342
Temperature | 0.439
Humidity 	| 0.367
Wind 	    | 0.428
--------------------------

Now make that feature root node, which has lowest ginni index, that means 
it is PURE, so Outlook is the root Node

Now the data has split according to the 3 unique values in it

This process goes on

So, decision tree building is over. 
We have built a decision tree by hand. 

"""


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  



"""
# open bill_authentication.csv 

Data Set Information:

Data were extracted from images that were taken from 
genuine and forged banknote-like specimens. 
For digitization, an industrial camera usually used for 
print inspection was used. 
The final images have 400x400 pixels. 
Due to the object lens and distance to the 
investigated object gray-scale pictures with a 
resolution of about 660 dpi were gained. 
Wavelet Transform tool were used to extract 
features from images.
"""

# This is a Classification problem

dataset = pd.read_csv("C:/Users/mohit/Desktop/Machine_Learning/Classification/bill_authentication.csv")  

#data analysis
dataset.shape

# Checking for Categorical Data
# Best part is that DT and RF works on the categorical data also
# We do not need to perform the Label encoding for it, Algo does it internally 

dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(100)


# Finding missing data
dataset.isnull().any(axis=0)


# Preparing the dataset
# This technique of dropping can be used when the label is in between features
features = dataset.drop('Class', axis=1)
print(features)
print(features.shape)

  
labels = dataset['Class']  
print(labels)
print(labels.shape)


# Train and test split
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.20)  

# Training and making predictions 
# We need to be careful in using DecissionTreeClassifier or DecissionTreeRegressor
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
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





# DecissionTree for Regression

"""
Problem Definition
The problem here is to predict the gas consumption (in millions of gallons) 
in 48 of the US states based on petrol tax (in cents), per capita income 
(dollars), paved highways (in miles) and the proportion of population with 
the driving license.
Govt to predict how much petrol to import
"""

import pandas as pd

# This is  a regression problem
dataset = pd.read_csv('C:/Users/mohit/Desktop/Machine_Learning/Classification/petrol_consumption.csv')  

#data analysis
dataset.shape


# Checking for Categorical Data
dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(10)


# Finding missing data
dataset.isnull().any(axis=0)


features = dataset.drop('Petrol_Consumption', axis=1)  
labels = dataset['Petrol_Consumption'] 

from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0) 

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
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



Q2. (Create a program that fulfills the following specification.)
Auto_mpg.txt

Here is the dataset about cars. The data concerns city-cycle fuel consumption in miles per gallon (MPG).

    Import the dataset Auto_mpg.txt
    Give the column names as "mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name" respectively
    Display the Car Name with highest miles per gallon value
    Build the Decision Tree and Random Forest models and find out which of the two is more accurate in predicting the MPG value
    Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 cylinders, having acceleration around 22.2 m/s due to it's 100 horsepower engine giving it a displacement of about 215. (Give the prediction from both the models)




Q4. tree_addhealth.csv
For this Code Challenge, The National Longitudinal Study of Adolescent to Adult Health (Add Health) data set, an ongoing (longitudinal) survey study that began in the mid-1990s is used. The project website URL is:

http://www.cpc.unc.edu/projects/addhealth/.

This large data set is available online from the University of North Carolina’s Carolina Population Center, http://www.cpc.unc.edu/projects/addhealth/data.

 

Import tree_addhealth.csv

 

The attributes are:

 

BIO_SEX: 1 = male 0 = female    

HISPANIC: 1=Yes,0=No    

WHITE : 1=Yes,0=No

BLACK : 1=Yes,0=No          

NAMERICAN: 1=Yes,0=No                      

ASIAN: 1=Yes,0=No                      

ALCEVR1: ever drank alcohol(1=Yes,0=No)   

marever1: ever smoked marijuana(1=Yes,0=No)    

cocever1: ever used cocaine(1=Yes,0=No)                

inhever1: ever used inhalants(1=Yes,0=No)             

cigavail: cigarettes available in home(1=Yes,0=No)

PASSIST: parents or public assistance(1=Yes,0=No)

EXPEL1: ever expelled from school(1=Yes,0=No)

TREG1: Ever smoked regularly(1=Yes,0=No)

Explanatory Variables:

Age

ALCPROBS1:alcohol problems 0-6

DEP1: depression scale

ESTEEM1: self esteem scale       

VIOL1:violent behaviour scale

DEVIANT1: deviant behaviour scale     

SCHCONN1: school connectedness scale       

GPA1: gpa scale  4 points)

FAMCONCT: family connectedness scale       

PARACTV:parent activities scale

PARPRES:parental presence scale
 
Build a classification tree model evaluating if an adolescent would smoke
 regularly or not based on: gender, age, (race/ethnicity) Hispanic, White, 
 Black, Native American and Asian, alcohol use, alcohol problems, marijuana 
 use, cocaine use, inhalant use, availability of cigarettes in the home, 
 depression, and self-esteem.

Build a classification tree model evaluation if an adolescent gets expelled 
or not from school based on their Gender and violent behavior.
Use random forest in relation to regular smokers as a target and explanatory 
variable specifically with Hispanic, White, Black, Native American and Asian.

(Please make confusion matrix and also check accuracy score for each 
and every section)


"""



# Skip from here onwards


"""

This code needs to be fixed on Mac and Windows
# Drawing the Decission Tree 
from sklearn import tree
#tree.plot_tree(classifier) #works with version 0.21.1


#to run below code: pip install graphviz
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("decision_tree") 

"""


"""


#classification
#from sklearn.metrics import classification_report
#print(classification_report(labels_test, labels_pred))


#Evaluation
#To evaluate performance of the regression algorithm, the commonly used metrics are mean absolute error, mean squared error, and root mean squared error.
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


#The mean absolute error for our algorithm is 54.7, 
which is less than 10 percent of the mean of all the values in the 
'Petrol_Consumption' column. This means that our algorithm did a fine 
prediction job.
"""




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



regressor.tree_.children_left #array of left children
regressor.tree_.children_right #array of right children
regressor.tree_.feature #array of nodes splitting feature
regressor.tree_.threshold #array of nodes splitting points
regressor.tree_.value #array of nodes values



from inspect import getmembers
print( getmembers( regressor.tree_ ) )

zip(dataset.columns[regressor.tree_.feature], regressor.tree_.threshold, regressor.tree_.children_left, regressor.tree_.children_right)

"""



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
"""