from datasets import load_dataset
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
 
print("📥 Loading dataset...")
dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

train_df = dataset["train"].to_pandas()
val_df   = dataset["validation"].to_pandas()
test_df  = dataset["test"].to_pandas()

# ✅ Drop rows where text is None or empty string
def clean_dataframe(df):
    df = df.dropna(subset=["text"])                # remove NaN
    df = df[df["text"].str.strip() != ""]          # remove empty
    return df

train_df = clean_dataframe(train_df)
val_df   = clean_dataframe(val_df)
test_df  = clean_dataframe(test_df)

print("✅ Dataset loaded")
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
print(train_df.head())

# Features/labels
X_train, y_train = train_df["text"], train_df["label"]
X_val, y_val     = val_df["text"], val_df["label"]
X_test, y_test   = test_df["text"], test_df["label"]

print("🚀 Training model...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=300)),
])

pipeline.fit(X_train, y_train)

train_acc = pipeline.score(X_train, y_train)
test_acc  = pipeline.score(X_test, y_test)

print(f"✅ Train Accuracy: {train_acc:.4f}")
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(pipeline, "models/sentiment_pipeline_3class.joblib")
print("✅ Model saved to models/sentiment_pipeline_3class.joblib")
