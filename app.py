import joblib
import pandas as pd
import streamlit as st

from features import add_features, FEATURES

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter a passenger's details and the model estimates their chance of surviving.")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2,
                          format_func=lambda x: {1: "1st (Upper)", 2: "2nd (Middle)", 3: "3rd (Lower)"}[x])
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    age = st.slider("Age", 0, 80, 30)
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"],
                         index=0 if sex == "male" else 2,
                         help="'Master' = young boy; 'Rare' = Dr, Rev, Col, nobility, etc.")
with col2:
    sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / children aboard", 0, 6, 0)
    fare = st.number_input("Fare (£)", 0.0, 520.0, 32.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"],
                            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "Pclass": str(pclass), "Sex": sex, "Age": age, "SibSp": sibsp, "Parch": parch,
        "Fare": fare, "Embarked": embarked, "Title": title, "Name": "",
    }])
    row = add_features(row)[FEATURES]
    prob = float(model.predict_proba(row)[0][1])

    st.progress(prob)
    if prob >= 0.5:
        st.success(f"Likely to SURVIVE — probability {prob:.1%}")
    else:
        st.error(f"Likely NOT to survive — probability {prob:.1%}")

st.caption("Model: Gradient Boosting trained on the Kaggle Titanic dataset (≈83–85% accuracy).")
