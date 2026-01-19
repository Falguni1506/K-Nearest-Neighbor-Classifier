import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"C:\Users\DELL\Downloads\diabetes.csv")

#Please use the path to your dataset file here.

df.head()


df.shape


x,y=df.shape
print(x,y)


df.describe()


df.isnull().sum()


df.Insulin.count()


w = 5
df.hist(bins=10, figsize=(20,15), color='green', alpha=0.6, hatch='X', rwidth=w);


X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


clf = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')


clf.fit(x_train,y_train)

pred = clf.predict(x_test)


print(confusion_matrix(pred, y_test))


print(accuracy_score(pred, y_test))


from sklearn.metrics import classification_report

print(classification_report(y_test, pred))





