import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report, confusion_matrix)
import seaborn as sns
import joblib

# Load data
df = pd.read_csv("spacex_launches.csv")

# Feature engineering
df = df.dropna(subset=["landing_success"])
df["landing_success"] = df["landing_success"].astype(int)
df["reused"] = df["reused"].fillna(False).astype(int)
df["gridfins"] = df["gridfins"].fillna(False).astype(int)
df["legs"] = df["legs"].fillna(False).astype(int)
df["flights"] = df["flights"].fillna(1)
df["landing_type"] = df["landing_type"].fillna("Unknown")
df = pd.get_dummies(df, columns=["landing_type"], drop_first=True)

FEATURES = ["reused", "gridfins", "legs", "flights"] + \
           [c for c in df.columns if c.startswith("landing_type_")]
TARGET = "landing_success"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="rbf", probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
best_score = 0
best_model = None
best_name = ""

print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)

for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

    if acc > best_score:
        best_score = acc
        best_model = clf
        best_name = name

print(f"\nBest Model: {best_name} (Accuracy: {best_score:.4f})")

# Save best model
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "features.pkl")
print("Model saved.")

# Confusion matrix
y_pred_best = best_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Success"], yticklabels=["Fail", "Success"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved.")
