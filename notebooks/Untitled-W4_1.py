
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer

df = pd.read_csv("C:\\Users\\abadj\\Documents\\GitHub\\CSC1171\\Week 1\\week4-DataIntro\\data\\week4_messy_survey.csv")

df_clean = df.copy().drop_duplicates()

# Standardise categorical labels with fixed, row-wise rules.
gender_map = {
    "f": "Female", "female": "Female", "m": "Male", "male": "Male",
    "non-binary": "Non-binary", "prefer not to say": "Prefer not to say",
}
education_map = {
    "high school": "High School", "bachelor": "Bachelor",
    "master": "Master", "phd": "PhD",
}

df_clean["gender"] = (
    df_clean["gender"].str.strip().str.lower().map(gender_map)
)
df_clean["education"] = (
    df_clean["education"].str.strip().str.lower().map(education_map)
)

# These values are fairly impossible rather than merely unusual. 
df_clean.loc[df_clean["age"] > 110, "age"] = np.nan
df_clean.loc[df_clean["annual_income_eur"] < 0, "annual_income_eur"] = np.nan

target = "satisfaction_score"


# A respondent ID is only an identifier. A complaint filed after the survey
# is known too late to use when predicting satisfaction at survey time.
leaky_cols = ["complaint_filed_after_survey"]
id_cols = ["respondent_id"]

X = df_clean.drop(columns=[target] + leaky_cols + id_cols)
y = df_clean[target]

# Keep 20% of the rows aside. random_state makes this split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sections 5-10 change Xtr/Xte. These copies keep X_train/X_test untouched
# for the all-in-one pipeline in section 11.
Xtr, Xte = X_train.copy(), X_test.copy()

# These are the columns where a median can be appropriate.
numeric_cols = [
    "age", "household_size", "annual_income_eur", "hours_online_per_week",
    "online_purchases_last_month",
]

# Learn one median per numeric column from TRAINING data only.
# The test set must not influence this choice.
medians = Xtr[numeric_cols].median()

# Reuse the training medians in both datasets.
Xtr[numeric_cols] = Xtr[numeric_cols].fillna(medians)
Xte[numeric_cols] = Xte[numeric_cols].fillna(medians)

# "Unknown" is a meaningful extra category for a nominal variable such as gender.
Xtr["gender"] = Xtr["gender"].fillna("Unknown")
Xte["gender"] = Xte["gender"].fillna("Unknown")

# Education is ordinal, so we use the most common TRAINING value instead.
education_mode = Xtr["education"].mode()[0]
Xtr["education"] = Xtr["education"].fillna(education_mode)
Xte["education"] = Xte["education"].fillna(education_mode)

# Minimal question: Based on the output, should the largest income be kept?
# It is unusual but still plausible, so we keep it.

def add_features(data):
    # Work on a copy, so the same function is safe for either split.
    result = data.copy()

    # Invalid date strings become missing values rather than causing an error.
    dates = pd.to_datetime(result["survey_date"], errors="coerce")
    result["survey_month"] = dates.dt.month
    result["survey_dayofweek"] = dates.dt.dayofweek

    # A household with the same income may have a different situation depending on its size.
    # clip(lower=1) prevents division by zero.
    result["income_per_person"] = (
        result["annual_income_eur"] / result["household_size"].clip(lower=1)
    )

    # clip(lower=0.1) prevents a zero number of online hours from causing an error.
    result["purchases_per_online_hour"] = (
        result["online_purchases_last_month"] / result["hours_online_per_week"].clip(lower=0.1)
    )

    # Age groups are categories, not numeric scores; they will be one-hot encoded later.
    result["age_group"] = pd.cut(
        result["age"], bins=[0, 29, 44, 64, np.inf],
        labels=["18-29", "30-44", "45-64", "65+"],
    ).astype("object")
    return result.drop(columns="survey_date")

# The recipe is fixed, so apply it identically to train and test data.
Xtr = add_features(Xtr)
Xte = add_features(Xte)
date_cols = ["survey_month", "survey_dayofweek"]
date_medians = Xtr[date_cols].median()
Xtr[date_cols] = Xtr[date_cols].fillna(date_medians)
Xte[date_cols] = Xte[date_cols].fillna(date_medians)
Xtr[["income_per_person", "purchases_per_online_hour", "age_group"]].head()

# This order is meaningful, so education is an ordinal variable.
edu_order = ["High School", "Bachelor", "Master", "PhD"]
education_encoder = OrdinalEncoder(categories=[edu_order])

# fit_transform learns the order from the training configuration and encodes training rows.
Xtr[["education"]] = education_encoder.fit_transform(Xtr[["education"]])

# transform applies exactly the same mapping to the test rows.
Xte[["education"]] = education_encoder.transform(Xte[["education"]])

# Gender and age group have no meaningful numeric order, so create separate 0/1 columns.
nominal_cols = ["gender", "age_group"]
nominal_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Learn which category columns exist from training data only.
train_nominal = pd.DataFrame(
    nominal_encoder.fit_transform(Xtr[nominal_cols]),
    columns=nominal_encoder.get_feature_names_out(nominal_cols), index=Xtr.index,
)

# An unseen test category is ignored rather than causing an error.
test_nominal = pd.DataFrame(
    nominal_encoder.transform(Xte[nominal_cols]),
    columns=nominal_encoder.get_feature_names_out(nominal_cols), index=Xte.index,
)

# Replace the original text columns with their new numeric columns.
Xtr = pd.concat([Xtr.drop(columns=nominal_cols), train_nominal], axis=1)
Xte = pd.concat([Xte.drop(columns=nominal_cols), test_nominal], axis=1)

# One-hot columns are already 0/1, so scale only continuous or count-based variables.
numeric_cols = [
    "age", "household_size", "annual_income_eur", "hours_online_per_week",
    "online_purchases_last_month", "survey_month", "survey_dayofweek",
    "income_per_person", "purchases_per_online_hour",
]

# The scaler learns the mean and standard deviation from training data only.
scaler = StandardScaler()
Xtr[numeric_cols] = scaler.fit_transform(Xtr[numeric_cols])

# The test set uses those same training values; it must not calculate new ones.
Xte[numeric_cols] = scaler.transform(Xte[numeric_cols])

def row_wise_features(data):
    """Create features using each row only; no dataset-wide values are learned here."""
    result = data.copy()

    # Turn one date column into two numeric calendar features.
    dates = pd.to_datetime(result["survey_date"], errors="coerce")
    result["survey_month"] = dates.dt.month
    result["survey_dayofweek"] = dates.dt.dayofweek

    # Create ratios before scaling, while values still use their original units.
    result["income_per_person"] = (
        result["annual_income_eur"] / result["household_size"].clip(lower=1)
    )
    result["purchases_per_online_hour"] = (
        result["online_purchases_last_month"] / result["hours_online_per_week"].clip(lower=0.1)
    )

    # Keep age bands as categories; the nominal pipeline will encode them.
    result["age_group"] = pd.cut(
        result["age"], bins=[0, 29, 44, 64, np.inf],
        labels=["18-29", "30-44", "45-64", "65+"],
    ).astype("object")
    return result.drop(columns="survey_date")

# Assign each column to the type of preprocessing it needs.
numeric_cols = [
    "age", "household_size", "annual_income_eur", "hours_online_per_week",
    "online_purchases_last_month", "survey_month", "survey_dayofweek",
    "income_per_person", "purchases_per_online_hour",
]
ordinal_cols = ["education"]
nominal_cols = ["gender", "age_group"]
edu_order = ["High School", "Bachelor", "Master", "PhD"]

# For numeric data: fill gaps with training medians, then put columns on a common scale.
numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# For education: fill gaps, then encode the supplied meaningful order.
ordinal_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(categories=[edu_order])),
])

# For nominal data: label missing values and create a 0/1 column for each category.
nominal_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

# Send each group of columns through its appropriate mini-pipeline.
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("ord", ordinal_pipeline, ordinal_cols),
    ("nom", nominal_pipeline, nominal_cols),
])

# First create features, then preprocess the resulting columns.
preprocessing_pipeline = Pipeline([
    ("features", FunctionTransformer(row_wise_features)),
    ("preprocess", preprocessor),
])

# fit_transform learns every required value from the training data and transforms it.
X_train_prepared = preprocessing_pipeline.fit_transform(X_train)

# transform applies those already-learned values to the unseen test data.
X_test_prepared = preprocessing_pipeline.transform(X_test)


col = "annual_income_eur"
print(f"median from ALL data   : {X[col].median():>10,.0f}")
print(f"median from TRAIN only : {X_train[col].median():>10,.0f}")
print(f"mean   from ALL data   : {X[col].mean():>10,.0f}")
print(f"mean   from TRAIN only : {X_train[col].mean():>10,.0f}")