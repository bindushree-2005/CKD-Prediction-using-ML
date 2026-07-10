import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================================
# Load Dataset
# ======================================================
df = pd.read_csv(r"archive\Training_CKD_dataset.csv")

print("Dataset Loaded Successfully!")

# ======================================================
# Convert Target to Binary
# Healthy Kidney = 0
# All CKD stages = 1
# ======================================================
# Convert Target to binary
df["Target"] = df["Target"].astype(str).str.strip()

df["Target"] = df["Target"].apply(
    lambda x: 0 if x == "Healthy Kidney" else 1
)

print(df["Target"].value_counts())

# ======================================================
# Separate Features and Target
# ======================================================
X = df.drop("Target", axis=1)
y = df["Target"]

# ======================================================
# Encode Categorical Columns
# ======================================================
label_encoders = {}

for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# ======================================================
# Handle Missing Values
# ======================================================
imputer = SimpleImputer(strategy="most_frequent")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ======================================================
# Split Dataset
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# Train Random Forest
# ======================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ======================================================
# Predict
# ======================================================
y_pred = model.predict(X_test)

# ======================================================
# Accuracy
# ======================================================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ======================================================
# Save Model
# ======================================================
joblib.dump(model, "ckd_random_forest.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\nModel Saved Successfully!")