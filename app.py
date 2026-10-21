import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info messages

from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# ---------- Page settings ----------
st.set_page_config(page_title="VillainNet", page_icon="🦹", layout="centered")

# Folder where app.py is saved, so the model files are found from anywhere
BASE = Path(__file__).parent


# ---------- Load the saved files (only once) ----------
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(str(BASE / "villainnet.keras"))  # trained VillainNet model
    scaler = joblib.load(BASE / "scaler.pkl")                           # the same fitted StandardScaler
    columns = joblib.load(BASE / "columns.pkl")                         # the 26 columns in training order
    return model, scaler, columns


model, scaler, columns = load_all()


# ---------- Requirement 2: Preprocessing (same as training) ----------
def preprocess(publisher, sex, identity, alive, eye, hair, appearances, year):
    # Start with one row of zeros that has the same 26 columns in the same order
    row = pd.DataFrame(0.0, index=[0], columns=columns)

    # One-hot encoding: put a 1 in the column of each chosen category
    row[f"publisher_{publisher}"] = 1.0
    row[f"sex_{sex}"] = 1.0
    row[f"identity_{identity}"] = 1.0
    row[f"alive_{alive}"] = 1.0
    row[f"eye_{eye}"] = 1.0
    row[f"hair_{hair}"] = 1.0

    # Apply log1p to the number of appearances (as in training)
    row["appearances"] = np.log1p(appearances)

    # Year of first appearance is used as it is
    row["year"] = year

    # Scale the values with the same fitted StandardScaler (transform only, no new fit)
    return scaler.transform(row)


# ---------- Page title ----------
st.title("🦸 VillainNet 🦹")
st.caption("A neural network that predicts whether a comic book character "
           "is a Hero or a Villain.")
st.divider()

# ---------- Requirement 1: Inputs (dropdowns for categories, number fields for numbers) ----------
left, right = st.columns(2)

with left:
    st.subheader("About the character")
    publisher = st.radio("Publisher", ["DC", "Marvel"], horizontal=True)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    identity = st.selectbox("Identity", ["Secret", "Public", "Other", "Unknown"])
    alive = st.radio("Alive or Deceased", ["Living", "Deceased"], horizontal=True)

with right:
    st.subheader("Looks and history")
    eye = st.selectbox("Eye colour", ["Black", "Blue", "Brown", "Green", "Unknown", "Other"])
    hair = st.selectbox("Hair colour", ["Black", "Blond", "Brown", "White", "Unknown", "Other"])
    appearances = st.number_input("Number of appearances", min_value=1, value=100, step=1)
    year = st.number_input("Year of first appearance", min_value=1900, max_value=2026,
                           value=1990, step=1)

st.divider()

# ---------- Requirement 3: Output (label + probability, for example "Villain: 72%") ----------
if st.button("Predict", type="primary", use_container_width=True):
    x = preprocess(publisher, sex, identity, alive, eye, hair, appearances, year)
    p_villain = float(model.predict(x, verbose=0)[0][0])   # probability that the character is a Villain

    if p_villain >= 0.5:
        st.error(f"Villain: {p_villain * 100:.0f}%")
    else:
        st.success(f"Hero: {(1 - p_villain) * 100:.0f}%")

    # Bar that shows the Villain probability
    st.progress(p_villain, text=f"Villain probability: {p_villain * 100:.0f}%")
