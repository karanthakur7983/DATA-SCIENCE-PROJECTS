# -*- coding: utf-8 -*-
"""Data science lab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SdnF-K8zHLV6KHEAlkpRSa9KBgZk180N
"""

import numpy as np

x = [10, 20, 30, 40, 50, 60]
y = [60, 4, 50, 90, 34, 54]


n = len(x)
mean_x = np.mean(x)
mean_y = np.mean(y)

print(mean_x)
print(mean_y)



number = 0
denom = 0
for i in range(n):
  number = number + (x[i] - mean_x) * (y[i] - mean_y)
  denom = denom + (x[i] - mean_x)**2
  m = number /denom
  c = mean_y - (m * mean_x)
  print(m)
  print(c)

import numpy as np
import matplotlib.pyplot as plt # Import matplotlib.pyplot and assign it to the alias plt

x = [10, 20, 30, 40, 50, 60]
y = [60, 4, 50, 90, 34, 54]

n = len(x)
mean_x = np.mean(x)
mean_y = np.mean(y)

# ... (rest of the code remains the same) ...

max_x =np.max(x) + 100
min_x =np.min(x) - 100
a = np.linspace(min_x, max_x, 1000)
# print(a)
b = m*a+c
# print(b)
plt.scatter(x,y,color = 'red') # Now plt is defined and can be used
plt.plot(a,b,color = 'blue')
plt.show()

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df = pd.read_excel("/content/excel file.xlsx")
print(df)

X=df[["weight(x)","height(y)"]].values
Y=df["class"].values
print(X)
print(Y)

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

Knn = KNeighborsClassifier(n_neighbors=7)
Knn.fit(X_train,Y_train)
df=np.array([57,170])
Y_predict=(Knn.predict([df]))
print(Y_predict)

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("/content/diabetes.csv")
print(df)

y=df["Outcome"].values
x=df[["Pregnancies","Glucose","BloodPressure","Insulin"]].values
print(y)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

Knn = KNeighborsClassifier(n_neighbors=7)
Knn.fit(x_train,y_train)
df=np.array([57,170,120,80])
y_predict=(Knn.predict([df]))
print(y_predict)

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("/content/user_behavior_dataset.csv")
print(df)

y=df["User Behavior Class"].values
x=df[["Device Model","Operating System","App Usage Time (min/day)","Screen On Time (hours/day)","Battery Drain (mAh/day)","Number of Apps Installed","Data Usage (MB/day)","Age","Gender"]].values
print(y)
print(x)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['Device Model']=le.fit_transform(df['Device Model'])
df['Operating System']=le.fit_transform(df['Operating System'])
df['Gender']=le.fit_transform(df['Gender'])

print(df.head)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)
print(y_train)
print(x_train)
print(y_test)

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv("/content/user_behavior_dataset.csv")

# Apply Label Encoding to the categorical features
le = preprocessing.LabelEncoder()
df['Device Model'] = le.fit_transform(df['Device Model'])
df['Operating System'] = le.fit_transform(df['Operating System'])
df['Gender'] = le.fit_transform(df['Gender'])

# Create the feature matrix (x) and target vector (y) AFTER Label Encoding
y = df["User Behavior Class"].values
x = df[["Device Model", "Operating System", "App Usage Time (min/day)", "Screen On Time (hours/day)", "Battery Drain (mAh/day)", "Number of Apps Installed", "Data Usage (MB/day)", "Age", "Gender"]].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the KNeighborsClassifier model
Knn = KNeighborsClassifier(n_neighbors=7)
Knn.fit(x_train, y_train)

# Make predictions (ensure 'df' for prediction has the correct features and encoded values)
df_predict = np.array([57, 170, 120, 80, 100, 50, 2000, 25, 1])  # Example, adjust values as needed
y_predict = Knn.predict([df_predict])
print(y_predict)

from sklearn.linear_model import LogisticRegression as logisticregression
logres= logisticregression()
logres.fit(x_train,y_train)
y_pred=logres.predict(x_test)
print(y_pred)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
le = LabelEncoder()
df['Device Model'] = le.fit_transform(df['Device Model'])
df['Operating System'] = le.fit_transform(df['Operating System'])
df['Gender'] = le.fit_transform(df['Gender'])
y = df["User Behavior Class"].values
x = df[["Device Model", "Operating System", "App Usage Time (min/day)", "Screen On Time (hours/day)", "Battery Drain (mAh/day)", "Number of Apps Installed", "Data Usage (MB/day)", "Age", "Gender"]].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train)
print(y_train)
print(x_train)
print(y_test)
logres = logisticregression()
logres.fit(x_train, y_train)
y_pred = logres.predict(x_test)

import pandas as pd
import numpy as np
df = pd.read_csv("/content/diabetes.csv")
print(df)

x = df[["Pregnancies","Glucose","BloodPressure","Insulin","DiabetesPedigreeFunction","Age"]].values
y = df["Outcome"].values
print(x)
print(y)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix (y_test,y_pred)
print(cnf_matrix)

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

clf = DecisionTreeClassifier()
clf= clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

plt.figure(figsize=(35,23))
plot_tree(clf,filled=True)
plt.show()