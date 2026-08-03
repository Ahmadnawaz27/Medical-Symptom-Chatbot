import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from symptom_extractor import extract_symptoms, SYMPTOM_KEYWORDS
from sklearn.model_selection import cross_val_score, StratifiedKFold


# Load dataset
df = pd.read_csv("Symptoms.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates()
df = df.dropna(subset=["text", "label"])

symptom_columns = list(SYMPTOM_KEYWORDS.keys())

# Build binary symptom feature for each text description
rows = []

for text in df["text"]:
    detected = extract_symptoms(text)
    row = {symptom: 1 if symptom in detected else 0 for symptom in symptom_columns}
    rows.append(row)

X = pd.DataFrame(rows)
y = df["label"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print(f"5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")


# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model and symptom columns
joblib.dump(model, "symptom_model.pkl")
joblib.dump(symptom_columns, "symptom_columns.pkl")

print("\nsymptom_model.pkl saved successfully.")
print("symptom_columns.pkl saved successfully.")