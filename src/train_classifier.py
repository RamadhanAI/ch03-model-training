
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, precision_recall_curve
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os

# Load and preprocess data
df = pd.read_csv('data/creditcard.csv')
X = df.drop(columns=['Class'])
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_probs = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred_thresh = (y_probs > threshold).astype(int)

print("Classification Report at Threshold 0.3")
print(classification_report(y_test, y_pred_thresh))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_thresh))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")

# Log to MLflow
mlflow.set_experiment("Fraud Detection CLI")
with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("threshold", threshold)
    mlflow.log_metric("precision", precision_score(y_test, y_pred_thresh))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_thresh))
    mlflow.sklearn.log_model(model, "model")

# Save precision-recall curve
prec, rec, thresh = precision_recall_curve(y_test, y_probs)
plt.figure()
plt.plot(thresh, prec[:-1], label='Precision')
plt.plot(thresh, rec[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/precision_recall_curve.png")
