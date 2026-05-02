"""Data loader module for loading and validating datasets"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Load and validate country datasets"""
    
    def __init__(self, data_path='countries of the world.csv'):
        """Initialize data loader with path to CSV file"""
        self.data_path = Path(data_path)
    
    def load_data(self):
        """Load countries dataset and return DataFrame"""
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} countries with {len(df.columns)} features")
        return df
    
    def clean_column_names(self, df):
        """Strip whitespace from column and country names"""
        df.columns = df.columns.str.strip()
        if 'Country' in df.columns:
            df['Country'] = df['Country'].str.strip()
        if 'Region' in df.columns:
            df['Region'] = df['Region'].str.strip()
        return df
    
    def convert_columns_to_float(self, df):
        """Convert string columns with commas to float"""
        columns_to_convert = [
            'Pop. Density (per sq. mi.)',
            'Coastline (coast/area ratio)',
            'Net migration',
            'Infant mortality (per 1000 births)',
            'Literacy (%)',
            'Phones (per 1000)',
            'Arable (%)',
            'Crops (%)',
            'Other (%)',
            'Birthrate',
            'Deathrate',
            'Agriculture',
            'Industry',
            'Service',
            'Climate'
        ]
        
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float, errors='ignore')
        
        return df
    
    def get_missing_values_summary(self, df):
        """Get summary of missing values"""
        missing_values_percentage = df.isnull().sum() / len(df) * 100
        missing_values_df = pd.DataFrame({
            'Column': missing_values_percentage.index,
            'Missing_Percentage': missing_values_percentage.values
        })
        missing_values_df = missing_values_df[missing_values_df['Missing_Percentage'] > 0].sort_values(
            by='Missing_Percentage', ascending=False
        )
        return missing_values_df
    
    def display_info(self, df):
        """Print dataset info"""
        print(f"\n📊 Dataset Shape: {df.shape}")
        print("\n🔍 Data Types and Missing Values:")
        df.info()
        print("\n📈 Statistical Summary:")
        print(df.describe())
        
        missing_df = self.get_missing_values_summary(df)
        if len(missing_df) > 0:
            print("\n⚠️  Missing Values:")
            print(missing_df)
        else:
            print("\n✓ No missing values found!")
        
        return df


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    df = loader.clean_column_names(df)
    df = loader.convert_columns_to_float(df)
    loader.display_info(df)
