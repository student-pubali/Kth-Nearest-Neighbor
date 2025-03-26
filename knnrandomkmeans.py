import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# KMeans Implementation
class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for _ in range(self.max_iter):
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            self.centroids = self.move_centroids(X, cluster_group)
            if np.all(old_centroids == self.centroids):
                break
        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        for row in X:
            distances = [np.linalg.norm(row - centroid) for centroid in self.centroids]
            cluster_group.append(np.argmin(distances))
        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        for cluster in np.unique(cluster_group):
            new_centroids.append(X[cluster_group == cluster].mean(axis=0))
        return np.array(new_centroids)

# KNN Implementation
class kNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        print("Training Done")

    def get_neighbors(self, test_point):
        distance = []
        for i, train_point in enumerate(self.X_train):
            dist = np.linalg.norm(test_point - train_point)
            distance.append((i, dist))
        distance.sort(key=operator.itemgetter(1))
        return distance[:self.k]

    def predict(self, X_test):
        return [self.classify(self.get_neighbors(test_point)) for test_point in X_test]

    def classify(self, distances):
        labels = [self.Y_train[i[0]] for i in distances]
        return Counter(labels).most_common(1)[0][0]

    def weighted_classify(self, distances):
        weights = {}
        for i, dist in distances:
            label = self.Y_train[i]
            weight = 1 / (dist + 1e-5)  # Avoid division by zero
            weights[label] = weights.get(label, 0) + weight
        return max(weights, key=weights.get)

# Load Dataset
df = pd.read_csv("iris.csv")
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KMeans Clustering
km = KMeans(n_clusters=3, max_iter=100)
y_means = km.fit_predict(X_train)

# Random Sampling
sample_size = int(math.sqrt(len(X_train)))
random_indices = random.sample(range(len(X_train)), sample_size)
X_sampled = X_train[random_indices]
Y_sampled = Y_train[random_indices]

# Test Different K Values
n_samples = int(math.sqrt(len(X_train)))
simple_knn_acc_kmeans, weighted_knn_acc_kmeans = [], []
simple_knn_acc_random, weighted_knn_acc_random = [], []

for k in range(1, n_samples + 1):
    # KMeans-based KNN
    knn_kmeans = kNearestNeighbors(k)
    knn_kmeans.fit(X_train, Y_train)
    simple_knn_acc_kmeans.append(accuracy_score(Y_test, knn_kmeans.predict(X_test)))

    knn_weighted_kmeans = kNearestNeighbors(k)
    knn_weighted_kmeans.fit(X_train, Y_train)
    weighted_knn_acc_kmeans.append(accuracy_score(Y_test, [knn_weighted_kmeans.weighted_classify(knn_weighted_kmeans.get_neighbors(x)) for x in X_test]))

    # Random Sampling-based KNN
    knn_random = kNearestNeighbors(k)
    knn_random.fit(X_sampled, Y_sampled)
    simple_knn_acc_random.append(accuracy_score(Y_test, knn_random.predict(X_test)))

    knn_weighted_random = kNearestNeighbors(k)
    knn_weighted_random.fit(X_sampled, Y_sampled)
    weighted_knn_acc_random.append(accuracy_score(Y_test, [knn_weighted_random.weighted_classify(knn_weighted_random.get_neighbors(x)) for x in X_test]))

# Plot Accuracy Comparisons
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_samples + 1), simple_knn_acc_kmeans, label='Simple KNN (KMeans)', marker='o')
plt.plot(range(1, n_samples + 1), weighted_knn_acc_kmeans, label='Weighted KNN (KMeans)', marker='x')
plt.plot(range(1, n_samples + 1), simple_knn_acc_random, label='Simple KNN (Random)', marker='o', linestyle='dashed')
plt.plot(range(1, n_samples + 1), weighted_knn_acc_random, label='Weighted KNN (Random)', marker='x', linestyle='dashed')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title("KMeans vs. Random Sampling for KNN")
plt.legend()
plt.show()

# Select Best Method
best_kmeans = max(max(simple_knn_acc_kmeans), max(weighted_knn_acc_kmeans))
best_random = max(max(simple_knn_acc_random), max(weighted_knn_acc_random))
print("Best Accuracy (KMeans-based KNN):", best_kmeans)
print("Best Accuracy (Random Sampling-based KNN):", best_random)

if best_kmeans > best_random:
    print("KMeans-based KNN is better!")
    best_model = kNearestNeighbors(n_samples)
    best_model.fit(X_train, Y_train)
else:
    print("Random Sampling-based KNN is better!")
    best_model = kNearestNeighbors(n_samples)
    best_model.fit(X_sampled, Y_sampled)

# Prediction using best model
def get_input():
    num_features = X_train.shape[1]
    return [float(input(f"Enter value for feature {i+1}: ")) for i in range(num_features)]

new_sample = get_input()
new_sample = scaler.transform(np.array(new_sample).reshape(1, -1))
predicted_class = best_model.predict(new_sample)[0]
print(f"The predicted class for the new sample is: {predicted_class}")
