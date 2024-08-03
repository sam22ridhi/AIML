import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from io import BytesIO

# Generate synthetic data
def generate_data(n_samples=100):
    X, y = make_classification(n_samples=n_samples, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    return X, y

# Prepare data
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# Define a function to plot 3D data and decision boundary
def plot_3d_logistic_regression(X, y, model, scaler):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolor='k')

    # Create a grid for decision boundary
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = np.zeros_like(x_grid)

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            sample = np.array([x_grid[i, j], y_grid[i, j], 0])
            sample = scaler.transform([sample])
            z_grid[i, j] = model.predict_proba(sample)[0, 1]

    # Plot decision boundary
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap='coolwarm')

    # Labels and title
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Probability of Class 1')
    ax.set_title('3D Logistic Regression Visualization')

    # Add a colorbar
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

    return fig

# Streamlit app
st.title("3D Logistic Regression Visualization")

st.write("This application demonstrates logistic regression with 3D visualization. The plot shows the data points and the decision boundary.")

fig = plot_3d_logistic_regression(X, y, model, scaler)
st.pyplot(fig)
