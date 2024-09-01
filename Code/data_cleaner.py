import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def remove_duplicates(self, df):
        return df.drop_duplicates()

    def handle_missing_values(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        return df

    def normalize_numeric_features(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df

    def clean_data(self, df):
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.normalize_numeric_features(df)
        return df

if __name__ == "__main__":
    # Example usage
    raw_data = pd.read_csv("data/raw/product_data.csv")
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(raw_data)
    cleaned_data.to_csv("data/processed/cleaned_product_data.csv", index=False)