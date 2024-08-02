# Install Necessary Libraries
!pip install pandas scikit-learn gradio matplotlib

# Load and Prepare the Titanic Dataset
import pandas as pd

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Data Cleaning and Preparation
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

X = data[['Pclass', 'Age', 'SibSp', 'Parch']]
y = data['Fare']

# Split the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Create a Function for Prediction and Visualization
import matplotlib.pyplot as plt
import gradio as gr

def predict_and_plot(Pclass, Age, SibSp, Parch):
    input_data = [[Pclass, Age, SibSp, Parch]]
    prediction = lin_reg.predict(input_data)[0]

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(y_test, lin_reg.predict(X_test), color='blue', label='Predicted vs Actual')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Fit')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()

    return prediction, fig

iface = gr.Interface(
    fn=predict_and_plot,
    inputs=[
        gr.Number(label="Pclass"),
        gr.Number(label="Age"),
        gr.Number(label="SibSp"),
        gr.Number(label="Parch")
    ],
    outputs=["number", "plot"],
    title="Titanic Fare Predictor with Visualization",
    description="Predict the fare of a Titanic passenger based on Pclass, Age, SibSp, and Parch, and visualize the prediction."
)

iface.launch()
