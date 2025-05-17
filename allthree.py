import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score



def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def extract_boundary_points_custom(X, y):
    boundary_points = []
    boundary_labels = []

    classes = np.unique(y)

    for cls in classes:
        X_cls = X[y == cls]
        X_other = X[y != cls]

        diffs = []

        for idx, point in enumerate(X_cls):
            same_class = np.delete(X_cls, idx, axis=0)
            d_intra = np.min([euclidean_distance(point, p) for p in same_class]) if len(same_class) > 0 else float('inf')
            d_inter = np.min([euclidean_distance(point, p) for p in X_other])

            d_diff = d_intra - d_inter
            diffs.append((point, d_diff))

        diffs.sort(key=lambda x: x[1])
        d_diff_values = [d[1] for d in diffs]
        threshold = np.mean(d_diff_values)

        for point, d_diff in diffs:
            if d_diff < threshold:
                boundary_points.append(point)
                boundary_labels.append(cls)

    return np.array(boundary_points), np.array(boundary_labels)

# ------------------- KMeans Clustering -------------------
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

# ------------------- kNN Implementation -------------------
class kNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distance = []
            for i, train_point in enumerate(self.X_train):
                dist = np.linalg.norm(test_point - train_point)
                distance.append((i, dist))
            distance.sort(key=operator.itemgetter(1))
            predictions.append(self.classify(distance[:self.k]))
        return predictions

    def classify(self, distance):
        labels = [self.Y_train[i[0]] for i in distance]
        return Counter(labels).most_common(1)[0][0]

# ------------------- Load and Preprocess Dataset -------------------
df = pd.read_csv("red-wine.csv")
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- Reduction Method 1: KMeans -------------------
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, max_iter=100)
cluster_labels = kmeans.fit_predict(X_train_scaled)

selected_X_kmeans, selected_Y_kmeans = [], []

for cluster in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    if len(cluster_indices) > 5:
        chosen_indices = np.random.choice(cluster_indices, 5, replace=False)
    else:
        chosen_indices = cluster_indices
    selected_X_kmeans.extend(X_train_scaled[chosen_indices])
    selected_Y_kmeans.extend(Y_train[chosen_indices])

selected_X_kmeans = np.array(selected_X_kmeans)
selected_Y_kmeans = np.array(selected_Y_kmeans)

# ------------------- Reduction Method 2: Boundary Points -------------------
boundary_X, boundary_Y = extract_boundary_points_custom(X_train_scaled, Y_train)

# ------------------- Reduction Method 3: Random Sampling -------------------
sample_size = len(boundary_X)
random_indices = np.random.choice(range(len(X_train_scaled)), sample_size, replace=False)
random_X = X_train_scaled[random_indices]
random_Y = Y_train[random_indices]

# ------------------- Accuracy Comparison -------------------
n_samples = int(math.sqrt(len(X_train)))
k_range = range(1, n_samples + 1)

accuracy_results = {
    "Full Dataset": [],
    "KMeans Reduced": [],
    "Boundary Points": [],
    "Random Sampling": []
}

for k in k_range:
    # Full dataset
    knn_full = kNearestNeighbors(k=k)
    knn_full.fit(X_train_scaled, Y_train)
    pred = knn_full.predict(X_test_scaled)
    accuracy_results["Full Dataset"].append(accuracy_score(Y_test, pred))

    # KMeans
    knn_kmeans = kNearestNeighbors(k=k)
    knn_kmeans.fit(selected_X_kmeans, selected_Y_kmeans)
    pred = knn_kmeans.predict(X_test_scaled)
    accuracy_results["KMeans Reduced"].append(accuracy_score(Y_test, pred))

    # Boundary
    knn_boundary = kNearestNeighbors(k=k)
    knn_boundary.fit(boundary_X, boundary_Y)
    pred = knn_boundary.predict(X_test_scaled)
    accuracy_results["Boundary Points"].append(accuracy_score(Y_test, pred))

    # Random
    knn_random = kNearestNeighbors(k=k)
    knn_random.fit(random_X, random_Y)
    pred = knn_random.predict(X_test_scaled)
    accuracy_results["Random Sampling"].append(accuracy_score(Y_test, pred))

# ------------------- Plotting All Accuracies -------------------
plt.figure(figsize=(12, 7))
for label, accuracies in accuracy_results.items():
    plt.plot(k_range, accuracies, label=label, marker='o')
plt.title("KNN Accuracy Comparison: Full vs Reduced Datasets")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
