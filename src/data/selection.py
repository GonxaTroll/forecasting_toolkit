""" selection.py
Methods for selecting data without considering as feature selection. Could be useful for
data preparation.
"""
import numpy as np
import pandas as pd
def get_columns_by_datatype(data: pd.DataFrame, column_type="categorical",
                            extra_columns = list(), exclude_columns = list()):
    extra_columns = data.columns.intersection(extra_columns)
    exclude_columns = data.columns.intersection(exclude_columns)
    if column_type == "categorical":
        column_type = [np.dtype("O")]
    elif column_type == "numerical":
        column_type = [np.dtype(f"{number_type}{precision}") for number_type in ["int", "float"]
                                                             for precision in [32, 64]]
        column_type += ["int8", "int16"]
    elif column_type == "date":
        column_type = [np.dtype('datetime64[ns]'), np.dtype('<M8[ns]')]
    columns = data.columns[np.isin(data.dtypes.values, column_type)]
    columns = columns.difference(exclude_columns)
    columns = extra_columns.union(columns)
    return columns


def get_dataset_by_datatype(data: pd.DataFrame, column_type="categorical",
                        extra_columns = list(), exclude_columns = list()):
    columns = get_columns_by_datatype(data, column_type, extra_columns, exclude_columns)
    return data[columns].copy(deep=True)


def select_categorical_with_many_categories(data: pd.DataFrame, cut_category_number: int=50,
                                            original_from_dummies: list = list()):
    cut_category_cols = []
    if len(original_from_dummies):
        for column in original_from_dummies:
            dummy_cols = data.columns[data.columns.str.startswith(column)]
            if len(dummy_cols) > cut_category_number:
                cut_category_cols += dummy_cols.tolist()
    else:
        for column in get_dataset_by_datatype(data, "categorical").columns:
            if len(data[column].value_counts()) > cut_category_number:
                cut_category_cols.append(column)
        print(f"Columns with over {cut_category_number} categories: {cut_category_cols}")
    return cut_category_cols
