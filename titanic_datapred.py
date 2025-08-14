# %%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic Dataset Analysis", page_icon="🚢", layout="centered")


st.title("Titanic Dataset Interactive Analysis & Prediction")

# --- Load data (cached) ---
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

train_df = load_data("/Users/sid/Desktop/Kaggle_Competitions/Titanic/Train.csv")
test_df = load_data("/Users/sid/Desktop/Kaggle_Competitions/Titanic/Test.csv")


# --- EDA (unchanged) ---
st.header("Exploratory Data Analysis (EDA)")
if st.checkbox("Show raw training data (first 10 rows)"):
    st.dataframe(train_df.head(10))

# Survival Rate Pie Chart
st.subheader("Survival Rate Distribution")
survival_counts = train_df['Survived'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(
    survival_counts,
    labels=['Did Not Survive', 'Survived'],
    autopct='%1.1f%%',
    colors=['#ff4c4c', '#4caf50'],
    startangle=90
)
ax1.axis('equal')
st.pyplot(fig1)

# Survival by Gender Bar Chart
st.subheader("Survival Rate by Gender")
fig2, ax2 = plt.subplots()
sns.barplot(x='Sex', y='Survived', data=train_df, ax=ax2)
ax2.set_ylabel("Survival Rate")
st.pyplot(fig2)

# Survival by Passenger Class - Stacked Bar Chart
st.subheader("Survival by Passenger Class")
pclass_survived = train_df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
fig3, ax3 = plt.subplots()
pclass_survived.plot(kind='bar', stacked=True, color=['#ff4c4c', '#4caf50'], ax=ax3)
ax3.set_xlabel("Passenger Class")
ax3.set_ylabel("Number of Passengers")
ax3.legend(['Did Not Survive', 'Survived'])
st.pyplot(fig3)

# Age distribution histogram with survival overlay
st.subheader("Age Distribution by Survival Status")
fig4, ax4 = plt.subplots()
train_df[train_df['Survived'] == 1]['Age'].plot(
    kind='hist', bins=30, alpha=0.6, color='#4caf50', label='Survived', ax=ax4
)
train_df[train_df['Survived'] == 0]['Age'].plot(
    kind='hist', bins=30, alpha=0.6, color='#ff4c4c', label='Did Not Survive', ax=ax4
)
ax4.set_xlabel("Age")
ax4.legend()
st.pyplot(fig4)

# --- Unified preprocessing function ---
def preprocess_for_model(df, is_train: bool = False, fit_columns=None):
    """
    Preprocesses a dataframe and returns feature DataFrame.
    If is_train=True, expects 'Survived' available; returns (X, y).
    If is_train=False, returns (X, ids) where ids is PassengerId if present else None.
    If fit_columns is provided, reindexes X to those columns (fill missing with 0).
    """
    d = df.copy()

    # Fill numeric missing values
    if 'Age' in d.columns:
        d['Age'].fillna(d['Age'].median(), inplace=True)
    if 'Fare' in d.columns:
        d['Fare'].fillna(d['Fare'].median(), inplace=True)

    # Fill Embarked
    if 'Embarked' in d.columns:
        d['Embarked'].fillna(d['Embarked'].mode()[0], inplace=True)

    # Keep PassengerId if present for output
    ids = d['PassengerId'].copy() if 'PassengerId' in d.columns else None

    # Drop text columns not used directly as features
    drop_cols = ['Cabin', 'Ticket', 'Name']
    # Drop PassengerId from features (keep ids separately)
    # We will drop PassengerId here from features as well
    drop_cols += ['PassengerId']
    for c in drop_cols:
        if c in d.columns:
            d.drop(columns=[c], inplace=True)

    # One-hot encode categorical variables we care about
    # Ensure consistent encoding: Sex, Embarked
    categorical_cols = []
    if 'Sex' in d.columns:
        categorical_cols.append('Sex')
    if 'Embarked' in d.columns:
        categorical_cols.append('Embarked')

    if categorical_cols:
        d = pd.get_dummies(d, columns=categorical_cols, drop_first=True)

    # If this is training data, separate label
    if is_train:
        if 'Survived' not in d.columns:
            raise ValueError("Training data must contain 'Survived'")
        y = d['Survived'].copy()
        X = d.drop(columns=['Survived'])
        # Align columns if fit_columns provided (unlikely for training)
        if fit_columns is not None:
            X = X.reindex(columns=fit_columns, fill_value=0)
        return X, y
    else:
        X = d
        if fit_columns is not None:
            X = X.reindex(columns=fit_columns, fill_value=0)
        return X, ids

# --- Preprocess training data (for training model) ---
processed_X_train, y_train = preprocess_for_model(train_df, is_train=True)

# --- Train model on processed features (this ensures names match predict-time) ---
model = RandomForestClassifier(random_state=42)
model.fit(processed_X_train, y_train)
st.success("Random Forest model trained successfully on processed training features.")

# --- Preprocess test data using training columns (guarantees matching feature names) ---
processed_X_test, test_ids = preprocess_for_model(test_df, is_train=False, fit_columns=processed_X_train.columns)

# --- Prediction on Test Set (with robust error handling) ---
st.header("Testing & Prediction")
try:
    predictions = model.predict(processed_X_test)
    # Show predictions with PassengerId if available
    if test_ids is not None:
        out_df = pd.DataFrame({'PassengerId': test_ids, 'Predicted_Survived': predictions})
        st.subheader("Predictions on Test Set")
        st.dataframe(out_df.head(50))
        # provide download button
        csv = out_df.to_csv(index=False)
        st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
    else:
        st.subheader("Predictions on Test Set")
        st.write(predictions)
except Exception as e:
    st.error("Prediction failed. Debug info below.")
    st.write("Exception:")
    st.write(str(e))
    st.write("Model seen these feature names during fit (sample):")
    st.write(list(processed_X_train.columns))
    st.write("Test features are (sample):")
    st.write(list(processed_X_test.columns))
    # Provide a helpful alignment attempt
    st.info("Attempting one more alignment: reindexing test to train columns and retrying...")
    processed_X_test_try = processed_X_test.reindex(columns=processed_X_train.columns, fill_value=0)
    try:
        predictions_try = model.predict(processed_X_test_try)
        st.success("Retry succeeded after reindexing!")
        if test_ids is not None:
            out_df_try = pd.DataFrame({'PassengerId': test_ids, 'Predicted_Survived': predictions_try})
            st.dataframe(out_df_try.head(50))
            st.download_button("Download adjusted predictions CSV", out_df_try.to_csv(index=False), "predictions_adjusted.csv", "text/csv")
        else:
            st.write(predictions_try)
    except Exception as e2:
        st.error("Retry also failed.")
        st.write(str(e2))

# --- Self-test (validation) using the SAME preprocessing pipeline ---
st.header("Self-Testing on Train Data (validation split)")

# Use processed_X_train and y_train (already preprocessed)
X_self = processed_X_train.copy()
y_self = y_train.copy()

# Split for validation
X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(X_self, y_self, test_size=0.2, random_state=42)

# Train fresh model for validation check (keeps the production model unchanged)
model_self_test = RandomForestClassifier(random_state=42)
model_self_test.fit(X_tr_split, y_tr_split)

y_val_pred = model_self_test.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_val_pred)
st.subheader("Validation Accuracy (train split)")
st.write(f"{val_accuracy * 100:.2f}%")

# Optional: Show confusion matrix (simple)
try:
    from sklearn.metrics import confusion_matrix
    import numpy as np
    cm = confusion_matrix(y_val_split, y_val_pred)
    st.subheader("Validation Confusion Matrix")
    st.write(pd.DataFrame(cm, index=['Actual_0','Actual_1'], columns=['Pred_0','Pred_1']))
except Exception:
    pass

# End of script
# %%
import os
print(os.getcwd())

# %%
