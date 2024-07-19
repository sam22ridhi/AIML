import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Title and Description
st.title("Linear Regression Visualization")
st.write("This application allows you to visualize Linear Regression on your dataset.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.write(data.head())
    
    # Select features for regression
    features = st.multiselect("Select features for regression", data.columns.tolist())
    target = st.selectbox("Select the target variable", data.columns.tolist())
    
    if len(features) > 0 and target:
        X = data[features].values
        y = data[target].values
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Linear Regression
        model = LinearRegression()
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)
        
        # Plotting
        st.write("Linear Regression Results:")
        fig, ax = plt.subplots()
        ax.scatter(X_scaled[:, 0], y, color='blue', label='Data points')
        ax.plot(X_scaled[:, 0], predictions, color='red', linewidth=2, label='Regression line')
        ax.set_xlabel(features[0])
        ax.set_ylabel(target)
        ax.legend()
        st.pyplot(fig)
        
        st.write("Model Coefficients:")
        st.write(model.coef_)
        st.write("Intercept:")
        st.write(model.intercept_)

# Instructions
st.write("Upload a CSV file, select the features and the target variable for regression, and visualize the results.")
