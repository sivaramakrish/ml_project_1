import pandas as pd
import os   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(file_name='melb_data.csv'):
    """Load the dataset from the specified CSV file in the 'input' folder."""
    # Construct the full path to the file
    file_path = os.path.join(os.getcwd(), 'input', file_name)
    
    # Debug print statement to verify the file path
    print(f"Loading data from: {file_path}")
    
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Debug: Check the shape of the loaded data
    print(f"Data shape after loading: {data.shape}")
    
    return data

def handle_missing_data(data):
    """Handle missing data using imputation."""
    # Impute numerical columns with the median
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    # Impute categorical columns with the mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer.fit_transform(data[categorical_cols])

    # Debug: Check the shape after missing data handling
    print(f"Data shape after handling missing data: {data.shape}")

    return data

def encode_categorical_variables(data):
    """Encode categorical variables using one-hot encoding."""
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    for col in categorical_cols:
        encoded_data = encoder.fit_transform(data[[col]])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.categories_[0])
        data = pd.concat([data, encoded_df], axis=1)
        data.drop(columns=[col], inplace=True)
    
    # Debug: Check the shape after encoding
    print(f"Data shape after encoding categorical variables: {data.shape}")
    
    return data

def split_data(data, target_column='Price'):
    """Split the dataset into features (X) and target (y), and then into train/test sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Debug: Check the shape of the data after splitting
    print(f"Data shape after splitting: X_train {X_train.shape}, X_valid {X_valid.shape}")
    
    return X_train, X_valid, y_train, y_valid

def preprocess_data(file_path):
    """Complete preprocessing pipeline."""
    # Load data
    data = load_data(file_path)
    
    # Handle missing values
    data = handle_missing_data(data)
    
    # Encode categorical variables
    data = encode_categorical_variables(data)
    
    # Split data into train and validation sets
    X_train, X_valid, y_train, y_valid = split_data(data)
    
    return X_train, X_valid, y_train, y_valid
