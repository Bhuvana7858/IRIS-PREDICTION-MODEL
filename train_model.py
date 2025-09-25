import pickle
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save model
with open("knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ KNN model saved as knn_model.pkl")
