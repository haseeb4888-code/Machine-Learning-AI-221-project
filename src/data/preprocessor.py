"""Data preprocessing module"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.ct = None
        self.numerical_cols = None
    
    def handle_missing_values(self, df):
        """Fill missing values with mean for numeric columns"""
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
        return df
    
    def remove_outliers(self, df, threshold=1.5):
        """Remove extreme outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"✓ Outliers removed. New shape: {df.shape}")
        return df
    
    def encode_categorical(self, df):
        """Encode categorical columns (Country, Region)"""
        df_encoded = df.copy()
        
        if 'Country' in df_encoded.columns:
            df_encoded['Country_Encoded'] = self.le.fit_transform(df_encoded['Country'])
        
        return df_encoded
    
    def scale_and_transform(self, df, fit=True):
        """Scale numerical features and one-hot encode region"""
        self.numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove Country_Encoded from numerical columns if present
        if 'Country_Encoded' in self.numerical_cols:
            self.numerical_cols.remove('Country_Encoded')
        
        # Create column transformer
        self.ct = ColumnTransformer([
            ('scale', StandardScaler(), self.numerical_cols),
            ('onehot', OneHotEncoder(sparse_output=False), ['Region'])
        ], remainder='drop')
        
        if fit:
            processed_data = self.ct.fit_transform(df)
        else:
            processed_data = self.ct.transform(df)
        
        # Create column names
        new_col_names = self.numerical_cols + self.ct.named_transformers_['onehot'].get_feature_names_out(['Region']).tolist()
        df_processed = pd.DataFrame(processed_data, columns=new_col_names)
        
        if 'Country_Encoded' in df.columns:
            df_processed['Country_Encoded'] = df['Country_Encoded'].values
        
        print(f"✓ Features scaled and transformed. Shape: {df_processed.shape}")
        return df_processed
    
    def preprocess(self, df):
        """Complete preprocessing pipeline"""
        print("\n🔧 Starting preprocessing pipeline...")
        df = self.handle_missing_values(df)
        print("✓ Missing values handled")
        
        df = self.encode_categorical(df)
        print("✓ Categorical features encoded")
        
        df_processed = self.scale_and_transform(df, fit=True)
        print("✓ Features scaled and transformed")
        
        return df_processed, df  # Return both processed and original for reference


if __name__ == "__main__":
    from src.data.loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_data()
    df = loader.clean_column_names(df)
    df = loader.convert_columns_to_float(df)
    
    preprocessor = DataPreprocessor()
    df_processed, df_original = preprocessor.preprocess(df)
    print("\n✓ Preprocessing complete!")
    print(f"Original shape: {df_original.shape}")
    print(f"Processed shape: {df_processed.shape}")
