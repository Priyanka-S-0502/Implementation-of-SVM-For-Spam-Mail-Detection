# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_scor

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by:Priyanka S 
RegisterNumber:212224040255


import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print (classification_report1)
x_test=cv.transform(x_test)
x_train
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
 
```

## Output:

<img width="973" height="50" alt="image" src="https://github.com/user-attachments/assets/8b956f48-42a1-41be-a3b2-5565a815c502" />

<img width="1045" height="301" alt="image" src="https://github.com/user-attachments/assets/c7d71816-2c85-479b-b02e-49d8c16eb994" />

<img width="548" height="342" alt="image" src="https://github.com/user-attachments/assets/87d2469a-4740-4f6f-bba5-a9b797d3df8d" />

<img width="1748" height="275" alt="image" src="https://github.com/user-attachments/assets/0b8903cf-e256-4f4c-bfc0-321f400c1f6e" />

<img width="1757" height="341" alt="image" src="https://github.com/user-attachments/assets/cb17ac50-29f4-4b4b-acbf-97e49e47b3ff" />

<img width="1318" height="60" alt="image" src="https://github.com/user-attachments/assets/c28e0012-3951-4820-a8d4-34c84b4cba5c" />

<img width="1359" height="64" alt="image" src="https://github.com/user-attachments/assets/e6be16d4-1b2e-44f3-bf59-577789bd1dcd" />

<img width="1583" height="84" alt="image" src="https://github.com/user-attachments/assets/98000122-bf75-439f-8e77-1e83367f4f4d" />

<img width="1579" height="261" alt="image" src="https://github.com/user-attachments/assets/0a8555f9-9d34-43d6-98dc-1bebbae21537" />










## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
