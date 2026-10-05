"""
DecodeLabs - AI Project 2: Data Classification Using AI
Pipeline: Input (Iris + scaling) -> Process (split + KNN) -> Output (confusion matrix + F1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------------
# 1. INPUT: load and understand the dataset
# ------------------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

print("Shape:", df.shape)                      # (150, 5)
print("\nFirst 5 rows:\n", df.head())
print("\nClass balance:\n", df["species"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("\nStatistics:\n", df.describe())

# ------------------------------------------------------------------
# 2. PROCESS: split FIRST, then scale (avoids data leakage)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 80% train / 20% test
    shuffle=True,        # randomize to remove order bias
    stratify=y,          # keep class proportions equal
    random_state=42,     # reproducible
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

#FEATURE SCALING
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # learn mean/std from TRAIN only
X_test_s = scaler.transform(X_test)         # apply same mean/std to TEST

# ------------------------------------------------------------------
# 3. Choose K (the "elbow") using error rate vs K
# ------------------------------------------------------------------
# Use 5-fold cross-validation on TRAINING data only, so the test set stays locked.
k_values = range(1, 31)
errors = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    acc = cross_val_score(knn, X_train_s, y_train, cv=5).mean()
    errors.append(1 - acc)

best_k = k_values[int(np.argmin(errors))]
print(f"\nBest K by lowest CV error: {best_k}")

plt.figure(figsize=(7, 4))
plt.plot(k_values, errors, marker="o")
plt.axvline(best_k, color="orange", linestyle="--", label=f"best K = {best_k}")
plt.xlabel("K value")
plt.ylabel("Error rate")
plt.title("Choosing K")
plt.legend()
plt.tight_layout()
plt.savefig("k_selection.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 4. Train the final model: Instantiate -> Fit -> Predict
# ------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=best_k)   # K chosen via cross-validation (slide example uses 5)
model.fit(X_train_s, y_train)     # train the model
print("\nModel trained successfully!")
predictions = model.predict(X_test_s)

# ------------------------------------------------------------------
# 5. OUTPUT: validate (accuracy alone is not enough)
# ------------------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:")
print(f"{accuracy:.4f}")

MacroF1 = f1_score(y_test, predictions, average="macro")
print("\nMacro F1:")
print(f"{MacroF1:.4f}")

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

classification_report = classification_report(y_test, predictions, target_names=iris.target_names)
print("\nClassification report:")
print(classification_report)


ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=iris.target_names, cmap="Blues"
)
plt.title(f"Confusion Matrix (KNN, K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 6. Test with completely new data (as the conclusion slide suggests)
# ------------------------------------------------------------------
new_flower = np.array([[5.9, 3.0, 5.1, 1.8]])   # sepal L/W, petal L/W in cm
new_flower_s = scaler.transform(new_flower)       # must scale the same way
pred = model.predict(new_flower_s)[0]
print("\nNew flower prediction:", iris.target_names[pred])
