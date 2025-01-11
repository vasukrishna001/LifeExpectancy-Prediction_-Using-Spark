import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Pre-trained Model
@st.cache(allow_output_mutation=True)
def loading_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Load the Random Forest model
model_path = "C:/Users/tarun/Documents/UI/models/random_forest.pkl"
rf_model = loading_model(model_path)

# Streamlit App
st.title("Life Expectancy Prediction")
st.sidebar.header("Enter Feature Values")

# Get the feature names from the model
training_features = rf_model.feature_names_in_  

# Create a dictionary to hold user inputs
user_inputs = {}

# generate  dynamically input fields for each feature
for feature in training_features:
    user_inputs[feature] = st.sidebar.number_input(f"Enter value for {feature}:", value=0.0)

# Converting user inputs to a DataFrame
df_input = pd.DataFrame([user_inputs])

st.write("Input Feature Values:")
st.dataframe(df_input)



# Make Predictions
try:
    predictions = rf_model.predict(df_input)
    st.write("Predicted Life Expectancy:")
    st.write(f"{predictions[0]:.2f}")

    # Simulated dataset for correlation analysis
    simulated_data = pd.DataFrame({feature: np.random.uniform(0, 100, 100) for feature in training_features})

    # Add a "Life Expectancy" column for correlation purposes
    simulated_data["Life Expectancy"] = np.random.uniform(50, 80, 100)

    # Display the correlation matrix 
    st.write("### Correlation Heatmap:")
    corr_matrix = simulated_data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation analysis")
    st.pyplot(fig)

    # Prediction Trends
    st.write("### Prediction Trends (e.g - Alcohol vs Life Expectancy):")
    alcohol_values = np.linspace(0, 10, 60)  
    trend_df = pd.DataFrame({feature: [user_inputs[feature]] * 60 for feature in training_features})
    trend_df["Alcohol"] = alcohol_values  
    predictions_trend = rf_model.predict(trend_df)

    fig, ax = plt.subplots()
    sns.lineplot(x=alcohol_values, y=predictions_trend, ax=ax)
    ax.set_title("Effect of Alcohol on Life Expectancy")
    ax.set_xlabel("Alcohol Consumption")
    ax.set_ylabel("Predicted Life Expectancy")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error during prediction: {e}")

