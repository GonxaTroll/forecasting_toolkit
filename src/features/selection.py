"""selection.py
This script contains functions to select features from a dataset.
"""
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from src.data.selection import get_dataset_by_datatype, get_columns_by_datatype
from src.data.analysis import compute_chi2
def get_highly_correlated_features(data: pd.DataFrame, threshold: float=0.9,
                                   **kwargs) -> list:
    """Calculates the correlation between each pair of variables and selects the ones with
       a correlation greater than the threshold.

    Args:
        data (pd.DataFrame): Numerical dataframe.
        threshold (float, optional): Value from which the correlation is considered.
                                     Defaults to 0.9.

    Returns:
        list: List of highly correlated features.
    """
    data = get_dataset_by_datatype(data, "numerical",
                                   exclude_columns = kwargs.get("exclude_columns"))
    linear_corrs = data.corr()
    linear_corrs = linear_corrs.unstack().reset_index()
    linear_corrs.columns = ["v1", "v2", "corr"]

    def reorder_corr_values(row: pd.Series):
        row = row.copy()
        if row["v1"] > row["v2"]:
            row["v1"], row["v2"] = row["v2"], row["v1"]
        return row

    linear_corrs = linear_corrs.apply(reorder_corr_values, axis=1)
    linear_corrs = linear_corrs.drop_duplicates()
    linear_corrs = linear_corrs[linear_corrs["v1"] != linear_corrs["v2"]]

    def get_acc_correlation(linear_corrs: pd.DataFrame):
        linear_corrs = pd.concat([linear_corrs[["v1","corr"]].rename(columns={"v1":"feature"}),
                                linear_corrs[["v2","corr"]].rename(columns={"v2":"feature"})
                                ])
        linear_corrs = linear_corrs.groupby(["feature"]).sum().reset_index()
        return linear_corrs

    features_to_delete = []
    while not linear_corrs[linear_corrs["corr"]>=threshold].empty:
        highly_correlated = linear_corrs[linear_corrs["corr"]>=threshold]
        highly_correlated = highly_correlated.melt(value_vars=["v1", "v2"])
        highly_correlated = highly_correlated["value"].value_counts().reset_index()
        highly_correlated = highly_correlated.rename(
            columns={highly_correlated.columns[0]:"feature",
                     highly_correlated.columns[1]:"count"})

        # Comparing the repeated features and getting the ones with most correlation sum
        accumulated_corrs = get_acc_correlation(linear_corrs)
        highly_correlated = highly_correlated.merge(accumulated_corrs, on=["feature"])
        highly_correlated = highly_correlated.sort_values(by=["count", "corr"],
                                                          ascending=[False, False])

        selected_feature = highly_correlated.iloc[0]["feature"]
        features_to_delete.append(selected_feature)

        # Removing the feature from the correlation matrix
        linear_corrs = linear_corrs[((linear_corrs["v1"]!=selected_feature) &
                                     (linear_corrs["v2"]!=selected_feature))]
    return features_to_delete


def remove_highly_correlated_features(data: pd.DataFrame, threshold: float = 0.9,
                                      **kwargs) -> pd.DataFrame:
    """Removes highly correlated features from a dataframe.

    Args:
        data (pd.DataFrame): Dataframe to remove highly correlated features.
        threshold (float, optional): Value from which the correlation is considered.
                                     Defaults to 0.9.

    Returns:
        pd.DataFrame: Dataframe with highly correlated features removed.
    """
    highly_correlated =\
        get_highly_correlated_features(data = data, threshold = threshold, **kwargs)
    data = data.drop(columns=highly_correlated)
    return data


def delete_constant_columns(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Deletes constant columns from a dataframe.

    Args:
        data (pd.DataFrame): Dataframe to delete constant columns.

    Returns:
        pd.DataFrame: Dataframe with constant columns deleted.
    """
    # Numerical features
    variance_threshold = 0
    numerical_data = get_dataset_by_datatype(data, "numerical")
    numerical_cols_delete = []
    if not numerical_data.empty:
        if kwargs.get("variance_threshold"):
            variance_threshold = kwargs["variance_threshold"]
        var_threshold = VarianceThreshold(threshold = variance_threshold)
        var_threshold.fit(numerical_data)
        numerical_variable_columns = var_threshold.get_feature_names_out().tolist()
        numerical_cols_delete = [x for x in numerical_data.columns
                                 if x not in numerical_variable_columns]

    # Categorical and datetime
    minimum_number_categories = 1
    categorical_columns = get_columns_by_datatype(data, "categorical")
    date_columns = get_columns_by_datatype(data, "date")
    categorical_columns = categorical_columns.union(date_columns)
    categorical_cols_delete = []
    if len(categorical_columns):
        if kwargs.get("minimum_number_categories"):
            minimum_number_categories = kwargs["minimum_number_categories"]
        for col in categorical_columns:
            if len(data[col].unique()) <= minimum_number_categories:
                categorical_cols_delete.append(col)
    cols_delete = numerical_cols_delete + categorical_cols_delete
    print(f"Number of constant columns deleted: {len(cols_delete)}")
    print(f"Columns deleted: {cols_delete}")
    return data.drop(columns=cols_delete)


def delete_null_columns(data: pd.DataFrame, proportion: float=0.7,
                        cols_to_exclude: list = None) -> pd.DataFrame:
    """Deletes columns with null values from a dataframe.

    Args:
        data (pd.DataFrame): Dataframe to delete null columns.
        proportion (float, optional): Proportion of null values to consider.
                                      Defaults to 0.7.
        cols_to_exclude (list, optional): List of columns to exclude.
                                          Defaults to [].

    Returns:
        pd.DataFrame: Dataframe with null columns deleted.
    """
    cols_to_exclude = cols_to_exclude if cols_to_exclude else []
    nulls = data.isnull().sum() / len(data)
    selected_cols = data.columns[nulls < proportion]
    null_cols =  data.columns[nulls >= proportion]
    cols_to_exclude = [x for x in cols_to_exclude if x not in selected_cols]
    null_cols = null_cols.difference(cols_to_exclude)
    print(f"{len(null_cols)} columns deleted: {null_cols}")
    return data.loc[:, selected_cols.tolist() + cols_to_exclude]


def select_by_chi2(data: pd.DataFrame, cols_to_exclude: list = None) -> pd.DataFrame:
    """Selects features by chi-square test.

    Args:
        data (pd.DataFrame): Dataframe to select features.
        cols_to_exclude (list, optional): List of columns to exclude.
                                          Defaults to [].

    Returns:
        pd.DataFrame: Dataframe with selected features.
    """
    cols_to_exclude = cols_to_exclude if cols_to_exclude else []
    chi2_df = compute_chi2(data)
    selected_variables = chi2_df[chi2_df["pvalue"]<0.05]["variable"].tolist()
    cols_to_exclude = [x for x in cols_to_exclude if x not in selected_variables]
    print(f"Not selected variables: {chi2_df[chi2_df['pvalue']>=0.05]['variable'].tolist()}")
    selected_variables += cols_to_exclude
    other_variables = data.columns.difference(chi2_df["variable"].tolist()).tolist()
    return data[other_variables+selected_variables]
