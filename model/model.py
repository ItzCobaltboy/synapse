import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import dump

# ============================
# Load MATLAB features
# ============================
data = np.loadtxt("miotracker_features.csv", delimiter=",")
X = data[:, :-1]
y = data[:, -1].astype(int)
print(f"Loaded: {X.shape}")
print(f"Classes: {np.unique(y)}")

# ============================
# CRITICAL FIX 1: Standardize FIRST (like MATLAB)
# ============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# CRITICAL FIX 2: MATLAB's 'auto' kernel scale
# ============================
# MATLAB computes kernel scale on STANDARDIZED data
# Formula: 1 / sqrt(P) where P = number of features
P = X_scaled.shape[1]
gamma_value = 1.0 / np.sqrt(P)
print(f"MATLAB-style gamma (1/sqrt({P})): {gamma_value}")

# ============================
# CRITICAL FIX 3: Quadratic = degree 2, NOT 4!
# CRITICAL FIX 4: coef0 = 1 (MATLAB default)
# CRITICAL FIX 5: Use One-vs-One
# ============================
base_svm = SVC(
    kernel='poly',
    degree=2,           # QUADRATIC means degree 2, not 4!
    gamma=gamma_value,  # MATLAB's auto scaling
    coef0=1.0,          # MATLAB default (not 0!)
    C=1.0,              # BoxConstraint
    probability=True,
    cache_size=1000
)

# MATLAB uses One-vs-One for multi-class
model = OneVsOneClassifier(base_svm)

# ============================
# Cross-validation matching MATLAB
# ============================
cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
fold_accuracies = []

print("\n10-Fold Cross-Validation:")
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)
    print(f"Fold {fold_idx}: {acc:.4f} ({acc*100:.2f}%)")

print(f"\nMean Accuracy: {np.mean(fold_accuracies):.4f} ({np.mean(fold_accuracies)*100:.2f}%)")
print(f"Std Accuracy: {np.std(fold_accuracies):.4f}")

# ============================
# Train final model on all data
# ============================
model.fit(X_scaled, y)

# Save both scaler and model
dump({'scaler': scaler, 'model': model}, "svm_quadratic_python.joblib")
print("\nSaved model → svm_quadratic_python.joblib")

# ============================
# Verification: Test prediction
# ============================
y_pred_all = model.predict(X_scaled)
train_acc = accuracy_score(y, y_pred_all)
print(f"Training set accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")