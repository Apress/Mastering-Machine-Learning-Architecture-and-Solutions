import pandas as pd


def validate_data(df):
    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Missing values detected!"

    # Validate data types
    assert df['age'].dtype == int, "Age column must be of type integer."

    # Ensure values fall within expected ranges
    assert df['age'].between(0, 120).all(), "Age values out of range."
