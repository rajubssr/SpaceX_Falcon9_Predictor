import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)

df = pd.read_csv('spacex_data.csv')

# Feature engineering
df = df.dropna(subset=['Outcome'])
df['Outcome'] = df['Outcome'].astype(int)

features = ['FlightNumber', 'PayloadMass', 'Flights', 'Block', 'ReusedCount',
            'GridFins', 'Reused', 'Legs']
df_model = df[features + ['Outcome']].dropna()

X = df_model[features]
y = df_model['Outcome']

# One-hot for boolean cols
for col in ['GridFins', 'Reused', 'Legs']:
    X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM':                 SVC(kernel='rbf', probability=True),
    'Decision Tree':       DecisionTreeClassifier(),
    'KNN':                 KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    results[name] = {'accuracy': acc, 'cv_score': cv_score}
    print(f"\n{name}:")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV Score (5-fold): {cv_score:.4f}")
    print(classification_report(y_test, y_pred))

# Best model
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_name]
print(f"\nBest Model: {best_name} with accuracy {results[best_name]['accuracy']:.4f}")

# Confusion matrix for best model
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix — {best_name}')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Model comparison bar chart
plt.figure(figsize=(8, 5))
names = list(results.keys())
accs  = [results[n]['accuracy'] for n in names]
sns.barplot(x=names, y=accs, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/model_comparison.png')
plt.close()

# Save best model and scaler
pickle.dump(best_model, open('best_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Best model and scaler saved!")
