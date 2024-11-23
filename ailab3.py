import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Lab task 1: Implement K-Nearest Neighbor (KNN) Algorithm
X_train = np.array([[0, 0], [1, 1], [2, 2], [1, 1], [0, 2]])
y_train = np.array([0, 1, 0, 1, 1]) 
query_instance = np.array([[1, 1]])
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predicted_class = knn.predict(query_instance)
print("Predicted class for the query instance:", "Play" if predicted_class == 1 else "No Play")

# Lab task 2: Confusion Matrix
X_test = np.array([[0, 0], [2, 2], [1, 1]]) 
y_test = np.array([0, 0, 1])
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
query_tissue = np.array([3, 7])
X_train_tissue = np.array([[1, 2], [4, 6], [2, 8], [3, 4], [6, 2]])
y_train_tissue = np.array([0, 1, 1, 0, 0])
distances = [euclidean_distance(query_tissue, train_point) for train_point in X_train_tissue]
print("\nEuclidean Distances:", distances)
sorted_indices = np.argsort(distances)
nearest_neighbors = y_train_tissue[sorted_indices[:3]]
print("Nearest Neighbors:", nearest_neighbors)
predicted_class_tissue = np.argmax(np.bincount(nearest_neighbors))
print("Predicted class for the new tissue paper:", "Class 1" if predicted_class_tissue == 1 else "Class 0")

# Home task Sample Dataset (Height, Weight) with labels (Healthy/Overweight)
X_train = np.array([[170, 65], [160, 75], [180, 85], [165, 70], [175, 80], [155, 60]])
y_train = np.array([0, 1, 1, 0, 1, 0])
X_test = np.array([[168, 72], [185, 90], [160, 65]])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("Test Data (Height, Weight) and Predictions:")
for i, (height, weight) in enumerate(X_test):
    print(f"Height: {height} cm, Weight: {weight} kg => Predicted: {'Healthy' if y_pred[i] == 0 else 'Overweight'}")
y_true = np.array([0, 1, 0])
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)