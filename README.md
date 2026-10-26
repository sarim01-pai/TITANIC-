---
title: Titanic Survival Predictor
emoji: 🚢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# Titanic Survival Predictor

Gradient Boosting model trained on the Kaggle Titanic dataset, served with Streamlit in a Docker Space.

- `train.py` — trains and saves `model.joblib`
- `features.py` — feature engineering (Title, FamilySize, IsAlone)
- `app.py` — Streamlit UI
