"""Data preprocessing utilities."""
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional

def process_methylation_data(
    data: Union[pd.DataFrame, str, Path],
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Process methylation beta values data.
    
    Args:
        data: Input methylation data as DataFrame or path to file
        columns: Optional list of CpG columns to use
        
    Returns:
        Processed DataFrame with standard CpG columns
    """
    # Load data if path provided
    if isinstance(data, (str, Path)):
        if str(data).endswith('.pkl'):
            df = pd.read_pickle(data)
        elif str(data).endswith('.csv'):
            df = pd.read_csv(data, index_col=0)
        elif str(data).endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data, index_col=0)
        else:
            raise ValueError("Unsupported file format. Use .pkl, .csv or .xlsx")
    else:
        df = data.copy()

    # Validate data format
    if df.empty:
        raise ValueError("Empty dataset provided")
    
    # Use provided columns or all numeric columns
    if columns is not None:
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df = df[columns]
    else:
        df = df.select_dtypes(include=[np.number])

    # Validate beta values
    if df.min().min() < 0 or df.max().max() > 1:
        raise ValueError("Beta values must be between 0 and 1")
        
    return df

def apply_pacmap(
    data: pd.DataFrame,
    pacmap_models: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply PaCMAP dimension reduction.
    
    Args:
        data: Methylation beta values
        pacmap_models: Dictionary of PaCMAP models
        
    Returns:
        2D and 5D PaCMAP embeddings
    """
    # Apply 2D PaCMAP
    embedding_2d = pacmap_models['2d'].transform(data.to_numpy(dtype='float16'))
    cols_2d = ['PaCMAP 1 of 2', 'PaCMAP 2 of 2']
    df_2d = pd.DataFrame(embedding_2d, columns=cols_2d, index=data.index)

    # Apply 5D PaCMAP
    embedding_5d = pacmap_models['5d'].transform(data.to_numpy(dtype='float16'))
    cols_5d = [f'PaCMAP {i+1} of 5' for i in range(5)]
    df_5d = pd.DataFrame(embedding_5d, columns=cols_5d, index=data.index)

    return df_2d, df_5d
