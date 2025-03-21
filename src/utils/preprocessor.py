import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess input data for prediction
    
    Args:
        data (Dict[str, Any]): Input data dictionary
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    try:
        # Convert input dictionary to DataFrame
        df = pd.DataFrame([data])
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
            
        # Fill categorical missing values with mode
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # Create derived features
        df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + \
                             df['BsmtFullBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0)
        df['Age'] = df['YrSold'] - df['YearBuilt']
        
        # Create quality score
        quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        df['QualityScore'] = df['OverallQual'].map(quality_mapping)
        
        logger.info("Successfully preprocessed input data")
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing input data: {str(e)}")
        raise 