import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = pd.read_csv("C:/Users/tarun/Documents/UI/output.csv")

# Split into features and target
X = data.drop("Life_Expectancy", axis=1) 
y = data['Life_Expectancy'] 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Save the model
os.makedirs("C:/Users/tarun/Documents/UI/models", exist_ok=True)
model_path = "C:/Users/tarun/Documents/UI/models/random_forest.pkl"
with open(model_path, "wb") as file:
    pickle.dump(rf_model, file)

print("Model saved successfully!")

