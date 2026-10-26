import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import add_features, NUM_COLS, CAT_COLS, FEATURES

train = add_features(pd.read_csv("train.csv"))
X, y = train[FEATURES], train["Survived"]

# Cast Pclass to str so it is one-hot encoded like the other categoricals
X = X.assign(Pclass=X["Pclass"].astype(str))

def make_pipe(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("clf", model)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "LogisticRegression": make_pipe(LogisticRegression(max_iter=1000)),
    "RandomForest": make_pipe(RandomForestClassifier(n_estimators=300, random_state=42)),
    "GradientBoosting": make_pipe(GradientBoostingClassifier(random_state=42)),
}
print("5-fold CV accuracy:")
for name, p in candidates.items():
    s = cross_val_score(p, X, y, cv=cv, scoring="accuracy")
    print(f"  {name:20s} {s.mean():.4f} (+/- {s.std():.4f})")

# Tune the best candidate (Gradient Boosting)
grid = GridSearchCV(
    make_pipe(GradientBoostingClassifier(random_state=42)),
    {"clf__n_estimators": [100, 200, 300],
     "clf__max_depth": [2, 3, 4],
     "clf__learning_rate": [0.03, 0.05, 0.1],
     "clf__subsample": [0.8, 1.0]},
    cv=cv, scoring="accuracy", n_jobs=-1,
)
grid.fit(X, y)
print("\nBest GB params:", grid.best_params_)
print(f"Best GB CV accuracy: {grid.best_score_:.4f}")

# Hold-out check for a classification report
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
holdout = make_pipe(GradientBoostingClassifier(**{k.split("__")[1]: v for k, v in grid.best_params_.items()}, random_state=42))
holdout.fit(Xtr, ytr)
print("\nHold-out accuracy:", round(accuracy_score(yva, holdout.predict(Xva)), 4))
print(classification_report(yva, holdout.predict(Xva), target_names=["Died", "Survived"]))

# Final model: refit on all training data and save
final = grid.best_estimator_
joblib.dump(final, "model.joblib")
print("Saved model.joblib")

# Predictions for the test set (Kaggle-style submission)
test = add_features(pd.read_csv("test.csv"))
Xt = test[FEATURES].assign(Pclass=test["Pclass"].astype(str))
pd.DataFrame({"PassengerId": test["PassengerId"],
              "Survived": final.predict(Xt)}).to_csv("submission.csv", index=False)
print("Saved submission.csv")

# Sanity check vs the gender-only baseline file
base = pd.read_csv("gender_submission.csv")
sub = pd.read_csv("submission.csv")
print("Agreement with gender_submission baseline:", round((base["Survived"] == sub["Survived"]).mean(), 4))
