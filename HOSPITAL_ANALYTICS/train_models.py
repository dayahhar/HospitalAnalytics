import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- Step 1: Load data ---
df = pd.read_csv("data/services_weekly.csv")

# --- Step 2: Encode categorical variables ---
df = pd.get_dummies(df, columns=["service", "event"], drop_first=True)

# --- Step 3: Define features and targets ---
features = ['week', 'month', 'available_beds', 'patients_request',
            'patient_satisfaction', 'staff_morale'] + \
           [col for col in df.columns if col.startswith('service_') or col.startswith('event_')]

X = df[features]
y_refused = df['patients_refused']
y_admitted = df['patients_admitted']

# --- Step 4: Split into train/test sets ---
X_trainA, X_testA, y_trainA, y_testA = train_test_split(X, y_refused, test_size=0.2, random_state=42)
X_trainB, X_testB, y_trainB, y_testB = train_test_split(X, y_admitted, test_size=0.2, random_state=42)

# --- Step 5: Train RandomForest models ---
modelA = RandomForestRegressor(n_estimators=200, random_state=42)
modelB = RandomForestRegressor(n_estimators=200, random_state=42)

modelA.fit(X_trainA, y_trainA)
modelB.fit(X_trainB, y_trainB)

# --- Step 6: Evaluate (optional) ---
print("Model A R^2:", modelA.score(X_testA, y_testA))
print("Model B R^2:", modelB.score(X_testB, y_testB))

# --- Step 7: Save models ---
joblib.dump(modelA, "models/model_refused.pkl")
joblib.dump(modelB, "models/model_admitted.pkl")

print("✅ Models saved successfully!")
