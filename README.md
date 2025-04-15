# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: YAMUNA M

RegisterNumber: 212223230248 

```python
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
![Image-1](https://github.com/user-attachments/assets/241e34b7-bc1e-49e7-b56a-2fd2217b399c)

![Image-2](https://github.com/user-attachments/assets/4e44f8e0-8bfd-4270-99f2-cf0c724acf5f)

![Image-3](https://github.com/user-attachments/assets/9919b46e-22ae-45b4-ac85-4d9c33b398cc)

![Image-4](https://github.com/user-attachments/assets/f5f578da-083e-483e-aa02-7ab2f7402187)

![Image-5](https://github.com/user-attachments/assets/2344bda3-5390-46cb-ad02-9e79f821a3f4)

![Image-6](https://github.com/user-attachments/assets/873d7e7d-487e-4ea7-a845-8fbc26345d0c)

![Image-7](https://github.com/user-attachments/assets/73031e50-bef6-4c9e-b722-d97fb0283ace)

![Image-8](https://github.com/user-attachments/assets/caec34ec-e2a2-4834-bc6e-70bed2487baf)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
