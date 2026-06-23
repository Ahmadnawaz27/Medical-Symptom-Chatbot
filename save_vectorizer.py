import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import utils

# Load dataset
df = pd.read_csv("Symptoms.csv")

# Clean dataset same way as App.py
df.drop("Unnamed: 0", axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# Keep same split logic as current app
train_data, test_data = train_test_split(df, test_size=0.15, random_state=42)

# Create and fit vectorizer once
vectorizer = utils.create_vectorizer()
vectorizer.fit(train_data["text"])

# Save fitted vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("vectorizer.pkl saved successfully.")