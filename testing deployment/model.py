import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load dataset
data = load_iris()
X, y = data.data, data.target

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# save model
joblib.dump(model, 'iris_model.joblib')
print("Model saved as 'iris_model.joblib'")