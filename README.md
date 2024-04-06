# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SINDHUJA P
RegisterNumber: 212222220047 
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## data.head()

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/2f372e59-6cd1-4421-874c-695cbe8a7b64)

## data.info()

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/9be0f352-955a-4e29-9013-277b69ce701d)


## data.isnull().sum()

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/48af3520-f70d-4f95-a247-1ff8d57f37b9)

## data value count

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/3b59ce89-59c7-4515-98fc-cd7f8c150443)

## data.head() for salary

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/5f477ef1-5616-44aa-80ed-4edbc1f0e16d)

## x.head()

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/440d3f7f-fb40-46cd-bdaf-a58626c7d508)

## accuracy value

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/9c8f38ac-f515-47e4-8ac9-c92beb49fd27)

## data prediction

![image](https://github.com/Sindhuja9585/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860624/f4c17c9d-4d89-4830-a320-d1df03ad33bb)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
