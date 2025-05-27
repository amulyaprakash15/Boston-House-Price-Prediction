# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('boston.csv')

# Drop missing values
df = df.dropna()

# Split features and target
X = df.drop(columns='MEDV')
y = df['MEDV']

# Split the data and train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------- Streamlit Frontend -----------------------

st.set_page_config(page_title="🏡 Boston House Price Predictor", layout="centered")

st.title("🏡 Boston House Price Prediction App")
st.markdown("Enter the details below to estimate the house price:")

# Dynamically create input fields
user_input = {}
for col in X.columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    user_input[col] = val

# Predict button
if st.button("🔮 Predict Price"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ${prediction:.2f}")
