import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report 
import joblib 
 
# Load dataset 
df = pd.read_csv(r"C:\Users\artik\pronostico_dataset (1).csv", delimiter=";") 
X = df[['age', 'systolic_bp', 'diastolic_bp', 'cholesterol']] 
y = df['prognosis'].map({'retinopathy': 1, 'no_retinopathy': 0}) 
 
# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Scale data 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
 
# Train model 
model = LogisticRegression(max_iter=1000) 
model.fit(X_train_scaled, y_train) 
 
# Save scaler and model 
joblib.dump(scaler, 'scaler.pkl') 
joblib.dump(model, 'logistic_model.pkl') 
 
# Evaluate 
y_pred = model.predict(X_test_scaled) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 
