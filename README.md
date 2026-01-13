#%% md

# 🧠 K-Nearest Neighbors (KNN) Algorithm

## ✅ What is KNN?
K-Nearest Neighbors (KNN) is a **supervised learning algorithm** used for **classification** and **regression**.
It works by finding the **K closest data points** (neighbors) to a given query point and making predictions based on those neighbors.

---

## 🔍 Key Steps in KNN:
1. **Choose K** (number of neighbors).
2. **Compute distance** between the query point and all training points.
3. **Sort distances** and select the K nearest neighbors.
4. **Vote or average**:
   - **Classification**: Majority vote among neighbors.
   - **Regression**: Average the neighbors' values.

---

## 📐 Distance Formula (Euclidean Distance)
For two points \( x = (x_1, x_2, \dots, x_n) \) and \( y = (y_1, y_2, \dots, y_n) \):

\[
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]

Other distance metrics:
- **Manhattan Distance**: \( d(x, y) = \sum |x_i - y_i| \)
- **Minkowski Distance**: Generalized form.

---

## 🛠 Python Implementation Example
```python
from sklearn.neighbors import KNeighborsClassifier

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
