"""Feature engineering module for creating derived features"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Create and engineer features from raw data"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.engineered_features = []
    
    def create_ratio_features(self, df):
        """Create ratio-based features from existing columns"""
        df_engineered = df.copy()
        
        # GDP-related ratios
        if 'GDP ($ per capita)' in df.columns and 'Infant mortality (per 1000 births)' in df.columns:
            df_engineered['GDP_to_Mortality'] = df['GDP ($ per capita)'] / (df['Infant mortality (per 1000 births)'] + 1)
            self.engineered_features.append('GDP_to_Mortality')
        
        # Population and area ratios
        if 'Population' in df.columns and 'Area (sq. mi.)' in df.columns:
            df_engineered['Pop_to_Area'] = df['Population'] / (df['Area (sq. mi.)'] + 1)
            self.engineered_features.append('Pop_to_Area')
        
        # Development indicators
        if 'Literacy (%)' in df.columns and 'Phones (per 1000)' in df.columns:
            df_engineered['Development_Index'] = (df['Literacy (%)'] + df['Phones (per 1000)']) / 2
            self.engineered_features.append('Development_Index')
        
        return df_engineered
    
    def create_demographic_features(self, df):
        """Create demographic-based features"""
        df_engineered = df.copy()
        
        # Birth-Death ratio
        if 'Birthrate' in df.columns and 'Deathrate' in df.columns:
            df_engineered['Birth_Death_Ratio'] = df['Birthrate'] / (df['Deathrate'] + 1)
            self.engineered_features.append('Birth_Death_Ratio')
        
        # Economic structure balance
        if 'Agriculture' in df.columns and 'Industry' in df.columns and 'Service' in df.columns:
            df_engineered['Economic_Balance'] = df['Service'] - df['Agriculture']
            self.engineered_features.append('Economic_Balance')
        
        return df_engineered
    
    def create_land_features(self, df):
        """Create land utilization features"""
        df_engineered = df.copy()
        
        # Total land utilization
        if 'Arable (%)' in df.columns and 'Crops (%)' in df.columns and 'Other (%)' in df.columns:
            df_engineered['Total_Land_Used'] = df['Arable (%)'] + df['Crops (%)'] + df['Other (%)']
            self.engineered_features.append('Total_Land_Used')
        
        return df_engineered
    
    def engineer_all_features(self, df):
        """Complete feature engineering pipeline"""
        print("\n🔧 Creating engineered features...\n")
        
        df = self.create_ratio_features(df)
        df = self.create_demographic_features(df)
        df = self.create_land_features(df)
        
        print(f"✓ Created {len(self.engineered_features)} new features:")
        for feat in self.engineered_features:
            print(f"  - {feat}")
        
        return df
