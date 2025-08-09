import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """
    Loads the diabetes dataset, performs preprocessing, and splits it for training.

    Args:
        file_path (str): The path to the diabetes dataset CSV file.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test, and the fitted scaler.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None

    # Replace zero values in key columns with the mean, as zero is not a valid
    # measurement for these parameters.
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan)

    df.fillna(df.mean(), inplace=True)

    # --- Advanced Feature Engineering ---
    # Create a new feature for BMI Category
    def get_bmi_category(bmi):
        if bmi < 18.5:
            return 0  # Underweight
        elif 18.5 <= bmi < 24.9:
            return 1  # Normal weight
        elif 24.9 <= bmi < 29.9:
            return 2  # Overweight
        else:
            return 3  # Obesity

    df['BMI_Category'] = df['BMI'].apply(get_bmi_category)

    # Define features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features to ensure they have similar distributions.
    # This is crucial for models like SVM to perform well.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
