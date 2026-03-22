import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier




df = pd.read_csv("data/Diabetes.csv")

print("Dataset Shape:", df.shape)


df = df.drop_duplicates()

print("Shape after duplicate removal:", df.shape)




df["age_group"] = pd.cut(
    df["age"],
    bins=[0,30,50,65,100],
    labels=[0,1,2,3]
).astype(int)

df["high_risk_comorbidity"] = (
    (df["hypertension"] == 1) &
    (df["heart_disease"] == 1)
).astype(int)

df["metabolic_risk"] = (
    (df["HbA1c_level"] > 6.5) |
    (df["blood_glucose_level"] > 126)
).astype(int)




df = df[df["smoking_history"] != "No Info"]


le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])

df["smoking_history"] = le_smoking.fit_transform(
    df["smoking_history"].astype(str)
)


for col in ["bmi","HbA1c_level","blood_glucose_level"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Shape after outlier removal:", df.shape)


#splitting features

X = df.drop("diabetes", axis=1)
y = df["diabetes"]


print("\nClass distribution:")
print(y.value_counts())


#train-test split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)




numeric_cols = ["age","bmi","HbA1c_level","blood_glucose_level"]

X_train[numeric_cols] = X_train[numeric_cols].astype(float)
X_test[numeric_cols] = X_test[numeric_cols].astype(float)




robust_scaler = RobustScaler()

X_train[["bmi","HbA1c_level","blood_glucose_level"]] = robust_scaler.fit_transform(
    X_train[["bmi","HbA1c_level","blood_glucose_level"]]
)

X_test[["bmi","HbA1c_level","blood_glucose_level"]] = robust_scaler.transform(
    X_test[["bmi","HbA1c_level","blood_glucose_level"]]
)


std_scaler = StandardScaler()

X_train[["age"]] = std_scaler.fit_transform(X_train[["age"]])
X_test[["age"]] = std_scaler.transform(X_test[["age"]])


#class imbalance handling

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

print("\nscale_pos_weight:", scale_pos_weight)



# Model


model = XGBClassifier(

    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist"

)



# Hyperparameter tuning


print("\nStarting hyperparameter tuning...")

param_dist = {

    "n_estimators":[200,300,400],
    "max_depth":[3,4,5,6],
    "learning_rate":[0.01,0.05,0.1],
    "subsample":[0.7,0.8,1],
    "colsample_bytree":[0.6,0.8,1]

}


search = RandomizedSearchCV(

    model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1

)


search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)



# Evaluation


preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:,1]

print("\nAccuracy:", accuracy_score(y_test,preds))
print("AUC:", roc_auc_score(y_test,probs))

print("\nClassification Report:")
print(classification_report(y_test,preds))



os.makedirs("models", exist_ok=True)

best_model.save_model("models/xgb_model.json")
joblib.dump(robust_scaler,"models/robust_scaler.joblib")
joblib.dump(std_scaler,"models/std_scaler.joblib")
joblib.dump(le_gender,"models/le_gender.joblib")
joblib.dump(le_smoking,"models/le_smoking.joblib")
joblib.dump(X.columns.tolist(),"models/feature_columns.joblib")

