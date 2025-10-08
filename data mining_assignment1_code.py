# =============================================
# task object：
#   Two classification models are used on the UCI Adult dataset（Logistic Regression 与 Random Forest）
#   Conduct training, testing and performance comparison.
# =============================================

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# ============================================================
# Data loading and preprocessing
# ============================================================

# Specify the data path
data_dir = Path("./")
train_path = data_dir / "adult.data"
test_path  = data_dir / "adult.test"

# Define column names
col_names = [
    "age","workclass","fnlwgt","education","education_num","marital_status",
    "occupation","relationship","race","sex","capital_gain","capital_loss",
    "hours_per_week","native_country","income"
]

# Define a function for reading adult.data / adult.test
def load_adult_file(path, is_test=False):
    # na_values：Will "" ? "" Marked as a missing value
    df = pd.read_csv(path, header=None, names=col_names, na_values=[" ?","?"], skipinitialspace=True)

    # The first line of adult.test is sometimes an explanatory line (not data) and needs to be skipped
    valid_labels = {"<=50K", ">50K", "<=50K.", ">50K."}
    if is_test and df.iloc[0]["income"] not in valid_labels:
        df = df.iloc[1:].reset_index(drop=True)

    # The income tag may end with '.' (for example, ">50K."), where the period is removed and Spaces are cleared
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    return df

# Read the training set and the test set, and remove the rows containing missing values
train_df = load_adult_file(train_path, is_test=False).dropna().reset_index(drop=True)
test_df  = load_adult_file(test_path, is_test=True).dropna().reset_index(drop=True)

# Split the feature (X) and the target label (y）
X_train = train_df.drop(columns=["income"])
y_train = (train_df["income"] == ">50K").astype(int)  # 将收入>50K编码为1，否则为0
X_test  = test_df.drop(columns=["income"])
y_test  = (test_df["income"] == ">50K").astype(int)

# Distinguish between numerical features and categorical features
numeric_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
categorical_features = [c for c in X_train.columns if c not in numeric_features]


# ============================================================
#  Model Setup
# ============================================================

# Logistic Regression：
#   - Standardize the numerical features（StandardScaler）
#   - Perform unique hot coding on category features（OneHotEncoder）
#   - Finally, use the LogisticRegression classifier
log_reg = Pipeline(steps=[
    ("prep", ColumnTransformer([
        ("num", StandardScaler(with_mean=False), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], sparse_threshold=0.3)),
    ("clf", LogisticRegression(max_iter=600, solver="lbfgs"))
])

# Random Forest ：
#   - The tree model is not affected by feature scaling, so the numerical features remain unchanged
rf = Pipeline(steps=[
    ("prep", ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])),
    ("clf", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
])


# ============================================================
# Model training and testing
# ============================================================

def fit_predict_metrics(model, X_tr, y_tr, X_te, y_te):
    """Auxiliary functions: Train the model, make predictions, and calculate performance metrics"""
    # 1. Record the training time
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t1 = time.time()
    tr_time = t1 - t0

    # 2. Record the predicted time
    t0 = time.time()
    y_pred = model.predict(X_te)
    t1 = time.time()
    pred_time = t1 - t0

    # 3. If the model supports predict_proba (probability output), then take the second column as the positive class probability
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_te)[:,1]
    else:
        y_proba = y_pred.astype(float)

    # 4. Calculate the main performance indicators
    metrics = dict(
        Accuracy = accuracy_score(y_te, y_pred),
        Precision = precision_score(y_te, y_pred),
        Recall = recall_score(y_te, y_pred),
        F1 = f1_score(y_te, y_pred),
        ROC_AUC = roc_auc_score(y_te, y_proba),
        Train_Time_s = tr_time,
        Predict_Time_s = pred_time
    )
    return metrics, y_pred, y_proba


# Train and evaluate Logistic Regression
m_lr, ypred_lr, yproba_lr = fit_predict_metrics(log_reg, X_train, y_train, X_test, y_test)

#Train and evaluate Random Forest
m_rf, ypred_rf, yproba_rf = fit_predict_metrics(rf, X_train, y_train, X_test, y_test)


# ============================================================
# Print the results and generate the report
# ============================================================

print("=== Logistic Regression ===")
print(m_lr)  # Output main indicators
print(classification_report(y_test, ypred_lr, digits=4))
print("Confusion Matrix (LR):")
print(confusion_matrix(y_test, ypred_lr))

print("\n=== Random Forest ===")
print(m_rf)
print(classification_report(y_test, ypred_rf, digits=4))
print("Confusion Matrix (RF):")
print(confusion_matrix(y_test, ypred_rf))


# ============================================================
# Save the result file
# ============================================================

# Save the indicator table
res_df = pd.DataFrame([
    {"Model":"Logistic Regression", **m_lr},
    {"Model":"Random Forest", **m_rf}
])
res_df.to_csv("in6227_assignment1_metrics.csv", index=False)

# Save detailed reports (including classification reports and confusion matrices)
with open("in6227_assignment1_reports.txt", "w") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(classification_report(y_test, ypred_lr, digits=4))
    f.write("\nConfusion Matrix (LR):\n")
    f.write(np.array2string(confusion_matrix(y_test, ypred_lr)))
    f.write("\n\n=== Random Forest ===\n")
    f.write(classification_report(y_test, ypred_rf, digits=4))
    f.write("\nConfusion Matrix (RF):\n")
    f.write(np.array2string(confusion_matrix(y_test, ypred_rf)))


# ============================================================
# Draw the ROC and Precision-Recall curves
# ============================================================

plt.figure()
RocCurveDisplay.from_predictions(y_test, yproba_lr, name="Logistic Regression")
RocCurveDisplay.from_predictions(y_test, yproba_rf, name="Random Forest")
plt.title("ROC Curve - Adult Test Set")
plt.savefig("roc_curves.png", bbox_inches="tight")
plt.close()

plt.figure()
PrecisionRecallDisplay.from_predictions(y_test, yproba_lr, name="Logistic Regression")
PrecisionRecallDisplay.from_predictions(y_test, yproba_rf, name="Random Forest")
plt.title("Precision-Recall Curve - Adult Test Set")
plt.savefig("pr_curves.png", bbox_inches="tight")
plt.close()

print("\n Done! Results saved as:")
print("- in6227_assignment1_metrics.csv")
print("- in6227_assignment1_reports.txt")
print("- roc_curves.png")
print("- pr_curves.png")


