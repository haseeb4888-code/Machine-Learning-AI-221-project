"""Tests for data loading and validation

This module contains tests to verify that:
- Data loads correctly from CSV
- All expected columns are present
- Data types are correct
- Missing values are within acceptable limits
- Data cleaning operations work properly
- The data is ready for training pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import DataLoader


class TestDataLoaderBasics:
    """Test basic data loading functionality"""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance"""
        return DataLoader('countries of the world.csv')
    
    def test_data_file_exists(self, loader):
        """Test that the data file exists"""
        assert loader.data_path.exists(), f"Data file not found at {loader.data_path}"
    
    def test_data_file_is_readable(self, loader):
        """Test that the data file is readable"""
        try:
            with open(loader.data_path, 'r') as f:
                f.read(1)
            assert True
        except Exception as e:
            pytest.fail(f"Data file is not readable: {e}")
    
    def test_data_loads_successfully(self, loader):
        """Test that data loads without errors"""
        df = loader.load_data()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_data_not_empty(self, loader):
        """Test that loaded data is not empty"""
        df = loader.load_data()
        assert len(df) > 0, "DataFrame is empty"
        assert len(df.columns) > 0, "DataFrame has no columns"


class TestDataColumns:
    """Test that the data has the expected columns"""
    
    @pytest.fixture
    def loaded_data(self):
        """Load and clean data for testing"""
        loader = DataLoader('countries of the world.csv')
        df = loader.load_data()
        df = loader.clean_column_names(df)
        return df
    
    EXPECTED_COLUMNS = [
        'Country', 'Region', 'Population', 'Area (sq. mi.)',
        'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
        'Net migration', 'Infant mortality (per 1000 births)',
        'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)',
        'Arable (%)', 'Crops (%)', 'Other (%)', 'Climate',
        'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service'
    ]
    
    def test_has_expected_columns(self, loaded_data):
        """Test that all expected columns are present"""
        for col in self.EXPECTED_COLUMNS:
            assert col in loaded_data.columns, f"Missing column: {col}"
    
    def test_column_count(self, loaded_data):
        """Test that the number of columns matches expected"""
        assert len(loaded_data.columns) == len(self.EXPECTED_COLUMNS), \
            f"Expected {len(self.EXPECTED_COLUMNS)} columns, got {len(loaded_data.columns)}"
    
    def test_no_unexpected_columns(self, loaded_data):
        """Test that there are no unexpected columns"""
        unexpected = set(loaded_data.columns) - set(self.EXPECTED_COLUMNS)
        assert len(unexpected) == 0, f"Unexpected columns found: {unexpected}"
    
    def test_country_column_exists(self, loaded_data):
        """Test that Country column exists and has data"""
        assert 'Country' in loaded_data.columns
        assert len(loaded_data['Country']) > 0
        assert loaded_data['Country'].notna().any()
    
    def test_region_column_exists(self, loaded_data):
        """Test that Region column exists and has data"""
        assert 'Region' in loaded_data.columns
        assert len(loaded_data['Region']) > 0
        assert loaded_data['Region'].notna().any()


class TestDataCleaning:
    """Test data cleaning operations"""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance"""
        return DataLoader('countries of the world.csv')
    
    def test_clean_column_names(self, loader):
        """Test that column names are cleaned (whitespace stripped)"""
        df = loader.load_data()
        df_cleaned = loader.clean_column_names(df)
        
        # Check that columns don't have leading/trailing spaces
        for col in df_cleaned.columns:
            assert col == col.strip(), f"Column '{col}' has leading/trailing spaces"
    
    def test_country_names_cleaned(self, loader):
        """Test that country names are cleaned"""
        df = loader.load_data()
        df = loader.clean_column_names(df)
        
        # Check that country names don't have extra whitespace
        for country in df['Country'].head(10):
            assert country == country.strip(), f"Country name '{country}' has extra whitespace"
    
    def test_region_names_cleaned(self, loader):
        """Test that region names are cleaned"""
        df = loader.load_data()
        df = loader.clean_column_names(df)
        
        # Check that region names don't have extra whitespace
        for region in df['Region'].dropna().head(10):
            assert region == region.strip(), f"Region name '{region}' has extra whitespace"


class TestDataTypeConversion:
    """Test data type conversions"""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance"""
        return DataLoader('countries of the world.csv')
    
    @pytest.fixture
    def processed_data(self, loader):
        """Load and process data for testing"""
        df = loader.load_data()
        df = loader.clean_column_names(df)
        df = loader.convert_columns_to_float(df)
        return df
    
    FLOAT_COLUMNS = [
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
    
    def test_numeric_columns_converted_to_float(self, processed_data):
        """Test that numeric columns are converted to float"""
        for col in self.FLOAT_COLUMNS:
            if col in processed_data.columns:
                # Check that column is numeric (float or int)
                assert pd.api.types.is_numeric_dtype(processed_data[col]), \
                    f"Column '{col}' is not numeric after conversion"
    
    def test_population_is_numeric(self, processed_data):
        """Test that Population column is numeric"""
        assert pd.api.types.is_numeric_dtype(processed_data['Population']), \
            "Population column is not numeric"
    
    def test_area_is_numeric(self, processed_data):
        """Test that Area column is numeric"""
        assert pd.api.types.is_numeric_dtype(processed_data['Area (sq. mi.)']), \
            "Area column is not numeric"
    
    def test_gdp_is_numeric(self, processed_data):
        """Test that GDP column is numeric"""
        assert pd.api.types.is_numeric_dtype(processed_data['GDP ($ per capita)']), \
            "GDP column is not numeric"
    
    def test_no_comma_strings_in_numeric_columns(self, processed_data):
        """Test that numeric columns don't have comma-separated strings"""
        for col in self.FLOAT_COLUMNS:
            if col in processed_data.columns:
                # Skip NaN values
                values = processed_data[col].dropna()
                if len(values) > 0:
                    # Convert to string and check for commas
                    string_values = values.astype(str)
                    assert not any(',' in str(v) for v in string_values), \
                        f"Column '{col}' still contains comma-separated values"


class TestDataValidation:
    """Test data validation and quality checks"""
    
    @pytest.fixture
    def processed_data(self):
        """Load and process data for testing"""
        loader = DataLoader('countries of the world.csv')
        df = loader.load_data()
        df = loader.clean_column_names(df)
        df = loader.convert_columns_to_float(df)
        return df
    
    def test_minimum_rows(self, processed_data):
        """Test that data has reasonable number of rows"""
        assert len(processed_data) >= 190, \
            f"Expected at least 190 countries, got {len(processed_data)}"
    
    def test_country_column_no_nulls(self, processed_data):
        """Test that Country column has no null values"""
        assert processed_data['Country'].notna().all(), \
            "Country column contains null values"
    
    def test_region_column_no_nulls(self, processed_data):
        """Test that Region column has no null values"""
        assert processed_data['Region'].notna().all(), \
            "Region column contains null values"
    
    def test_population_positive(self, processed_data):
        """Test that Population values are positive"""
        population = processed_data['Population'].dropna()
        assert (population > 0).all(), \
            "Population contains non-positive values"
    
    def test_area_positive(self, processed_data):
        """Test that Area values are positive"""
        area = processed_data['Area (sq. mi.)'].dropna()
        assert (area > 0).all(), \
            "Area contains non-positive values"
    
    def test_literacy_in_valid_range(self, processed_data):
        """Test that Literacy values are between 0-100"""
        literacy = processed_data['Literacy (%)'].dropna()
        if len(literacy) > 0:
            assert (literacy >= 0).all() and (literacy <= 100).all(), \
                "Literacy contains values outside 0-100 range"
    
    def test_missing_values_acceptable(self, processed_data):
        """Test that missing values are within acceptable limits"""
        missing_df = self._get_missing_values_summary(processed_data)
        
        # No column should have more than 50% missing values
        for idx, row in missing_df.iterrows():
            assert row['Missing_Percentage'] <= 50, \
                f"Column '{row['Column']}' has too many missing values ({row['Missing_Percentage']:.2f}%)"
    
    @staticmethod
    def _get_missing_values_summary(df):
        """Get summary of missing values"""
        missing_values_percentage = df.isnull().sum() / len(df) * 100
        missing_values_df = pd.DataFrame({
            'Column': missing_values_percentage.index,
            'Missing_Percentage': missing_values_percentage.values
        })
        return missing_values_df[missing_values_df['Missing_Percentage'] > 0].sort_values(
            by='Missing_Percentage', ascending=False
        )
    
    def test_no_duplicate_countries(self, processed_data):
        """Test that there are no duplicate country entries"""
        assert not processed_data['Country'].duplicated().any(), \
            f"Found duplicate country entries: {processed_data[processed_data['Country'].duplicated()]['Country'].tolist()}"
    
    def test_countries_unique(self, processed_data):
        """Test that the number of unique countries matches total rows"""
        assert len(processed_data['Country'].unique()) == len(processed_data), \
            "Mismatch between unique countries and total rows"


class TestDataPipeline:
    """End-to-end tests for the complete data loading pipeline"""
    
    def test_full_loading_pipeline(self):
        """Test the complete data loading pipeline"""
        loader = DataLoader('countries of the world.csv')
        
        # Load
        df = loader.load_data()
        assert df is not None
        initial_rows = len(df)
        
        # Clean
        df = loader.clean_column_names(df)
        assert len(df) == initial_rows, "Row count changed after cleaning"
        
        # Convert
        df = loader.convert_columns_to_float(df)
        assert len(df) == initial_rows, "Row count changed after conversion"
        
        # Validate
        assert df is not None
        assert len(df) > 0
        assert 'Country' in df.columns
        assert 'Region' in df.columns
    
    def test_pipeline_ready_for_training(self):
        """Test that data is ready for training pipeline"""
        loader = DataLoader('countries of the world.csv')
        df = loader.load_data()
        df = loader.clean_column_names(df)
        df = loader.convert_columns_to_float(df)
        
        # Should have numeric features for modeling
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        assert len(numeric_columns) > 10, \
            f"Not enough numeric features for training. Found {len(numeric_columns)}"
        
        # Should have sufficient rows
        assert len(df) >= 100, \
            f"Insufficient data for training. Got {len(df)} rows, need at least 100"
        
        # Should have minimal critical missing values
        critical_columns = ['Country', 'Region', 'Population', 'Area (sq. mi.)']
        for col in critical_columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            assert missing_pct < 5, \
                f"Column '{col}' has {missing_pct:.2f}% missing values"
