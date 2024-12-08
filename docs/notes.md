# Project Notes

## Step 1: Dataset Preprocessing
- Loaded data from `melb_data.csv` file.
- Handled missing values in numerical and categorical columns.
- Applied one-hot encoding for categorical variables.

## Step 2: Model Training
- Split the data into training and validation sets.
- Trained a random forest model on the data.
- Model training completed successfully.

## Step 3: Evaluation
- Evaluated model performance using MAE metric.
- Validation MAE: 161,455.39.

## Next Steps
- Explore other evaluation metrics like RMSE.
- Try different models (e.g., Gradient Boosting, XGBoost).

## Challenges
- Handling missing data in categorical columns required imputation strategy.
- Model training took longer due to large number of categorical variables after encoding.

