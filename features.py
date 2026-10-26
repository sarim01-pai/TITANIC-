import pandas as pd

RARE_TITLES = {"Lady", "the Countess", "Capt", "Col", "Don", "Dr", "Major",
               "Rev", "Sir", "Jonkheer", "Dona"}

def extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    if title in ("Mlle", "Ms"):
        return "Miss"
    if title == "Mme":
        return "Mrs"
    if title in RARE_TITLES:
        return "Rare"
    return title

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared by training and the Streamlit app."""
    df = df.copy()
    if "Title" not in df.columns:
        df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

NUM_COLS = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title"]
FEATURES = NUM_COLS + CAT_COLS
