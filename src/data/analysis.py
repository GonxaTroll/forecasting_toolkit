import pandas as pd
from scipy.stats import chi2_contingency
def compute_chi2(data: pd.DataFrame) -> pd.DataFrame:
    """Computes the chi-square test for categorical variables.

    Args:
        data (pd.DataFrame): Dataframe with categorical variables.

    Returns:
        pd.DataFrame: Dataframe with chi-square test results.
    """

    categorical_v = data.columns[(data.dtypes == "object") | (data.dtypes == "bool")]
    chi2_df = []
    for col in categorical_v:
        contingency = pd.crosstab(index=data["target"],
                                  columns = [data[col]])
        _, pvalue, _, _ = chi2_contingency(contingency.values)
        chi2_df.append([col, pvalue])

    chi2_df = pd.DataFrame(chi2_df, columns=["variable", "pvalue"])
    return chi2_df
