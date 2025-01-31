"""utils.py
Several utility functions for data manipulation.
"""
import pandas as pd
def reduce_column_size(data: pd.DataFrame) -> pd.DataFrame:
    """Reduces the size of the columns of a dataframe to the smallest possible.

    Args:
        data (pd.DataFrame): Initial dataframe.

    Returns:
        pd.DataFrame: Dataframe with reduced columns.
    """
    for col in data.columns:
        dtype = str(data[col].dtype)
        if "int" in dtype:
            _reduce_column_size_integer(data, col)
        elif "float" in dtype:
            _reduce_column_size_float(data, col)
    return data


def _reduce_column_size_integer(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reduces the size of an integer column to the smallest possible.

    Args:
        data (pd.DataFrame): Initial dataframe.
        col (str): Column to reduce.

    Returns:
        pd.DataFrame: Dataframe with reduced column.
    """
    max_abs_value = abs(data[col]).max()
    for exp in [8, 16, 32, 64]:
        if 2**(exp-1) >= max_abs_value:
            data[col] = data[col].astype(f"int{exp}")
            break
    return data


def _reduce_column_size_float(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reduces the size of a float column to the smallest possible.

    Args:
        data (pd.DataFrame): Initial dataframe.
        col (str): Column to reduce.

    Returns:
        pd.DataFrame: Dataframe with reduced column.
    """
    max_abs_value = abs(data[col]).max()
    for exp in [32, 64]:
        if 2**(exp-1) >= max_abs_value:
            data[col] = data[col].astype(f"float{exp}")
            break
    return data


def order_columns_alphabetically(data: pd.DataFrame, initial_columns: list = None) -> pd.DataFrame:
    """Orders the columns of a dataframe alphabetically. When initial_columns is provided,
       the columns are ordered after the initial_columns.

    Args:
        data (pd.DataFrame): Initial dataframe.
        initial_columns (list, optional): Columns to keep before the alphabetical order.
                                          Defaults to None.

    Raises:
        ValueError: initial_columns must be a list.

    Returns:
        pd.DataFrame: Dataframe with ordered columns.
    """
    initial_columns = [] if initial_columns is None else initial_columns
    if not isinstance(initial_columns, list):
        raise ValueError("initial_columns must be a list")
    other_columns = [x for x in data.columns.sort_values()
                     if x not in initial_columns]
    data = data[initial_columns + other_columns]
    return data
