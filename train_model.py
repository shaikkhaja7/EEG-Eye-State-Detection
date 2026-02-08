import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "EEG_Eye_State_Classification.csv"
MODEL_OUT = BASE_DIR / "users" / "EEGEye.pkl"

print("✅ BASE_DIR:", BASE_DIR)
print("✅ Dataset path:", DATASET_PATH)

# ---- Load dataset ----
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

# ---- Split X & y ----
X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Train model ----
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate ----
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("\n✅ Model trained successfully!")
print("✅ Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, pred))

# ---- Save model ----
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_OUT)

print("\n✅ Model saved to:", MODEL_OUT)
