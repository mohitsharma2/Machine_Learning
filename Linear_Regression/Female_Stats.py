"""
 (Create a program that fulfills the following specification.)
Female_Stats.Csv


Import The Female_Stats.Csv File

The Data Are From N = 214 Females In Statistics Classes At The University Of California At Davi.

Column1 = Student’s Self-Reported Height,

Column2 = Student’s Guess At Her Mother’s Height, And

Column 3 = Student’s Guess At Her Father’s Height. All Heights Are In Inches.

 

Build A Predictive Model And Conclude If Both Predictors (Independent Variables)
Are Significant For A Students’ Height Or Not
When Father’s Height Is Held Constant, 
The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
When Mother’s Height Is Held Constant, 
The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.

"""
import pandas as pd
import numpy as np

dataset=pd.read_csv("C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/Female_Stats.csv")

dataset.isnull().any(axis=0)

# 1) Build A Predictive Model And Conclude If Both Predictors (Independent Variables)
#Are Significant For A Students’ Height Or Not

features=dataset.iloc[:,[1,2]].values

labels=dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)



from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  

regressor.fit(features_train, labels_train)
print("Training_score=",regressor.score(features_train, labels_train)*100,"%") 

Pred = regressor.predict(features_test)
print("Test_score=",regressor.score(features_test, labels_test)*100,"%")
x=pd.DataFrame(zip(Pred, labels_test))
print(x)

regressor.predict([[75,67]])  # output=array([67.36960254])
regressor.predict([[76,68]])  # output=array([68.07868334])


# 2)  When Father’s Height Is Held Constant, 
#The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.

import pandas as pd
import numpy as np

dataset=pd.read_csv("C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/Female_Stats.csv")

dataset.isnull().any(axis=0)

features=dataset.iloc[:,[2]].values

labels=dataset.iloc[:,[0]].values

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  


regressor.fit(features_train, labels_train)
 
Pred = regressor.predict(features_test)

x=pd.DataFrame(zip(Pred, labels_test))
print(x)

regressor.predict([[64]])    
#Output: array([[62.0402137]])

regressor.predict([[65]])
#Output: array([[62.55586254]])

regressor.predict([[66]])
#Output: array([[63.07151138]])


# 3) When Mother’s Height Is Held Constant, 
#The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.

import pandas as pd
import numpy as np

dataset=pd.read_csv("C:/Users/mohit/Desktop/Machine_Learning/Linear_Regression/Female_Stats.csv")

dataset.isnull().any(axis=0)

features=dataset.iloc[:,-2].values
labels=dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=41)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
features_train=features_train.reshape(171,1)

regressor.fit(features_train,labels_train)

Pred = regressor.predict(features_test)

x=pd.DataFrame(zip(Pred, labels_test))
print(x)

regressor.predict([[62]])    
#Output: array([64.07066542])

regressor.predict([[63]])
#Output: array([64.56216346])

regressor.predict([[64]])
#Output: array([65.05366151])
