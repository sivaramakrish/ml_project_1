# Project Progress Documentation

## Step 1: Dataset Preprocessing

1. **Load the Dataset**
   - **Description:** Load the dataset from a CSV file (`melb_data.csv`) located in the `input` folder.
   - **Function:** `load_data(file_name)`
   - **Code:**
     ```python
     data = load_data(file_name='melb_data.csv')
     ```
   - **Issue:** File loading was successful, no errors encountered at this stage.
   - **Debug Print:**
     ```python
     print(f"Loading data from: {file_path}")
     ```

2. **Handle Missing Data**
   - **Description:** Impute missing data for numerical columns using median and for categorical columns using the most frequent value (mode).
   - **Function:** `handle_missing_data(data)`
   - **Code:**
     ```python
     data = handle_missing_data(data)
     ```
   - **Issue:** The process was successful with no errors during data imputation.
   - **Debug Print:**
     ```python
     print(f"Data shape after handling missing data: {data.shape}")
     ```

3. **Encode Categorical Variables**
   - **Description:** One-hot encode categorical variables.
   - **Function:** `encode_categorical_variables(data)`
   - **Code:**
     ```python
     data = encode_categorical_variables(data)
     ```
   - **Issue:** The categorical columns were successfully encoded, but the number of columns grew drastically due to encoding.
   - **Debug Print:**
     ```python
     print(f"Data shape after encoding categorical variables: {data.shape}")
     ```

4. **Data Splitting**
   - **Description:** Split the dataset into training and validation sets using an 80-20 split.
   - **Function:** `split_data(data, target_column='Price')`
   - **Code:**
     ```python
     X_train, X_valid, y_train, y_valid = split_data(data)
     ```
   - **Issue:** Data split was successful. However, the number of features (columns) increased significantly due to one-hot encoding.
   - **Debug Print:**
     ```python
     print(f"Data shape after splitting: X_train {X_train.shape}, X_valid {X_valid.shape}")
     ```

---

## Step 2: Model Training

1. **Training the Model**
   - **Description:** Train a Random Forest model on the preprocessed dataset.
   - **Code:**
     ```python
     model = RandomForestRegressor(n_estimators=100, random_state=42)
     model.fit(X_train, y_train)
     ```
   - **Issue:** Model training completed without errors. 
   - **Debug Print:**
     ```python
     print("Model training completed.")
     ```

2. **Model Evaluation**
   - **Description:** Evaluate the trained model on the validation set using the Mean Absolute Error (MAE) metric.
   - **Code:**
     ```python
     y_pred = model.predict(X_valid)
     mae = mean_absolute_error(y_valid, y_pred)
     print(f"Validation MAE: {mae}")
     ```
   - **Issue:** The model evaluation was completed successfully with an MAE value of `161,455.39`.
   - **Debug Print:**
     ```python
     print(f"Validation MAE: {mae}")
     ```

3. **Save the Model**
   - **Description:** Save the trained model to a file.
   - **Code:**
     ```python
     joblib.dump(model, 'models/random_forest_model.pkl')
     ```
   - **Issue:** Model was successfully saved.

---

## Step 3: Errors Encountered and Resolutions

### Error 1: `TypeError: handle_missing_data() takes 1 positional argument but 2 were given`
   - **Cause:** The `handle_missing_data` function was originally written to handle only one dataset, but it was mistakenly called twice in the `preprocess_data` function.
   - **Fix:** Removed the extra call to `handle_missing_data()` for both `X_train` and `X_valid` separately.
   - **Resolved Code:**
     ```python
     X_train, X_valid, y_train, y_valid = preprocess_data(file_path)
     ```

### Error 2: `ValueError: at least one array or dtype is required`
   - **Cause:** This error occurred during the imputation of categorical data when trying to pass empty data into the imputer.
   - **Fix:** Ensure that the categorical columns are non-empty before applying the imputer.
   - **Resolved Code:**
     ```python
     categorical_cols = data.select_dtypes(include=['object']).columns
     if len(categorical_cols) > 0:
         data[categorical_cols] = imputer.fit_transform(data[categorical_cols])
     ```

### Error 3: Unexpected Growth in Columns Due to One-Hot Encoding
   - **Cause:** After one-hot encoding categorical variables, the dataset expanded by many columns due to the encoding of each category into separate columns.
   - **Fix:** No fix was needed as this is expected behavior when using one-hot encoding for many categorical variables.
   - **Solution:** The increase in columns was understood and handled, though it caused an increase in model complexity.

---

## Step 4: Git Version Control

1. **Commit Changes**
   - After making all the necessary changes, the following files were added and committed to Git:
     - `dataset.py` (with updated handling of missing data and encoding functions)
     - `train.py` (with model training and evaluation steps)
     - `notes.md` (documentation of steps and errors)

   - **Git Commands:**
     ```bash
     git add .
     git commit -m "Completed preprocessing and model training. Added progress notes."
     git push origin main
     ```

2. **Push Changes to GitHub**
   - Changes were pushed to the GitHub repository to maintain version control and share the progress with collaborators.

---

## Next Steps

1. **Model Improvement**
   - Try other models (e.g., Gradient Boosting, XGBoost) to compare performance.
   - Tune hyperparameters of the Random Forest model to improve accuracy.

2. **Evaluate on Different Metrics**
   - Explore other evaluation metrics like RMSE or R² to get more insights into the model's performance.

3. **Refactor and Modularize Code**
   - Refactor the code for better modularity and reusability.
   - Split the code into separate modules for data preprocessing, model training, and evaluation.

---

This document tracks the key steps taken during the development, along with the challenges encountered and their resolutions. You can continue to update it as new tasks or challenges arise.
