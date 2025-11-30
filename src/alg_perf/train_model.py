import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("sorting_data.csv")
print("Loaded rows:", len(df))
print("\nfastest_label counts:\n", df['fastest_label'].value_counts())

# Quick checks
print("\nNumeric summary (first 10 numeric columns):")
print(df.select_dtypes(include='number').describe().T[['count','mean','std','min','max']].head(10))

# If only one class exists, stop and tell user to re-run benchmark
if df['fastest_label'].nunique() < 2:
    print("\nERROR: Only one label found in 'fastest_label'. Re-run benchmark.py with more diverse generators or fix tie logic.")
    raise SystemExit(1)

# Prepare X,y
drop_cols = ['pattern','seed','fastest_label','time_quick','time_merge','time_heap']
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols)
y = df['fastest_label']

# convert any non-numeric to numeric if needed
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        try:
            X[c] = pd.to_numeric(X[c])
        except:
            X[c] = X[c].astype('category').cat.codes

# Remove constant columns
const_cols = [c for c in X.columns if X[c].nunique() <= 1]
if const_cols:
    print("Dropping constant columns:", const_cols)
    X = X.drop(columns=const_cols)

# Stratify only if possible
if y.value_counts().min() < 2:
    strat = None
    print("Warning: cannot stratify because a class has <2 samples.")
else:
    strat = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)
print("Train label counts:\n", y_train.value_counts())
print("Test label counts:\n", y_test.value_counts())

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

imp = clf.feature_importances_
for name, val in sorted(zip(X.columns, imp), key=lambda x: x[1], reverse=True):
    print(f"{name}: {val:.6f}")

joblib.dump(clf, "model.pkl")
print("Saved model.pkl")
