import numpy as np
import pandas as pd
from kNearestNeighbors import kNearestNeighbors
data = pd.read_csv('C:/Users/PUBALI KARAN/Knn/Social_Network_Ads.csv')

# print(data.head())

X = data.iloc[:,2:4].values
Y = data.iloc[:,-1].values

# print(X.shape)
# print(Y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# print(X_train.shape)
# print(X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(X_train)

# Create an object of the knn class
knn = kNearestNeighbors(k=3)

knn.fit(X_train,Y_train)

# knn.predict(np.array([60,100000]).reshape(1,2))

def predict_new():
    age=int(input("Enter the age"))
    salary=int(input("Enter the salary"))
    X_new = np.array([[age],[salary]]).reshape((1,2))

    X_new=scaler.transform(X_new)

    result = knn.predict(X_new)
    if result==0:
        print("Will not purchase")
    else:
        print("Will purchase")

predict_new()
