import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('plant_growth_data.csv')

# Separate features and target variable
X = data.drop('Growth_Milestone', axis=1)
y = data['Growth_Milestone']

# Define categorical and numerical features
categorical_features = ['Soil_Type', 'Water_Frequency', 'Fertilizer_Type']
numerical_features = ['Sunlight_Hours', 'Temperature', 'Humidity']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Full pipeline with regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predict and evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

def get_user_input():
    # Collect user inputs
    Soil_Type = input("Enter soil type: ")
    Sunlight_Hours = float(input("Enter hours of sunlight: "))
    Water_Frequency = input("Enter water frequency: ")
    Fertilizer_Type = input("Enter fertilizer type: ")
    Temperature = float(input("Enter temperature: "))
    Humidity = float(input("Enter humidity: "))
    
    # Create a DataFrame for user inputs
    user_data = pd.DataFrame({
        'Soil_Type': [Soil_Type],
        'Sunlight_Hours': [Sunlight_Hours],
        'Water_Frequency': [Water_Frequency],
        'Fertilizer_Type': [Fertilizer_Type],
        'Temperature': [Temperature],
        'Humidity': [Humidity]
    })
    return user_data

# Get user input and predict
user_input = get_user_input()
user_input_preprocessed = model.named_steps['preprocessor'].transform(user_input)
user_prediction = model.named_steps['regressor'].predict(user_input_preprocessed)
print(f'Predicted Growth Milestone: {user_prediction[0]}')

# Plotting the results
plt.figure(figsize=(10, 6))

# Scatter plot for actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.7, label='Test Data')
plt.xlabel('Actual Growth Milestone')
plt.ylabel('Predicted Growth Milestone')
plt.title('Actual vs Predicted Growth Milestone')
plt.legend()

# Highlight the user input prediction
plt.scatter([y_test.mean()], [user_prediction[0]], color='red', marker='x', s=100, label='User Input Prediction')
plt.legend()

# Plot diagonal line (ideal prediction line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--')

plt.show()
