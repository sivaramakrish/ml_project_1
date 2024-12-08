from dataset import preprocess_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib  # For saving the model

# Step 1: Preprocess the data
file_path = 'melb_data.csv'  # Path to your dataset
# Run the preprocessing pipeline
X_train, X_valid, y_train, y_valid = preprocess_data(file_path)

# Print the first few rows of the processed data (X_train)
print("First few rows of X_train:")
print(X_train.head())  # Print the head of X_train

# Step 2: Train the model
print("Training the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Initialize model
model.fit(X_train, y_train)  # Train the model
print("Model training completed.")

# Step 3: Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_valid)  # Predict on validation set
mae = mean_absolute_error(y_valid, y_pred)  # Calculate mean absolute error
print(f"Validation MAE: {mae}")

# Step 4: Save the trained model
model_path = 'models/random_forest_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
