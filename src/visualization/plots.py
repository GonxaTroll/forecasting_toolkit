import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def plot_null_percent(data: pd.DataFrame, proportion: float = 0.7):
    """Plots the percentage of null values in each column of a dataframe.

    Args:
        data (pd.DataFrame): Dataframe from which to compute the percentage of null values.
        proportion (float, optional): Proportion of null values to show as threshold.
                                      Defaults to 0.7.
    """
    plt.figure(figsize=(35,5))
    nulls = (data.isnull().sum() / len(data)).sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    nulls = pd.DataFrame(nulls).reset_index()
    nulls.columns = ["variable", "null_percentage"]
    ax = sns.barplot(data=nulls, x="variable", y="null_percentage")
    plt.xticks(rotation=45)

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.001,
                f'{height:.3%}',
                ha="center")
    if max(nulls["null_percentage"]) >= proportion:
        plt.hlines(y = proportion, linestyles="dashed", xmin = -0.5, xmax = len(nulls)-0.5,
                   colors = "red")
    # return fig
    plt.show()


def plot_chi2(data: pd.DataFrame, cols_to_exclude: list = []):
    for col in categorical_v:
        counts = data[col].value_counts(normalize=True).reset_index()
        plt.figure(figsize=(25,5))
        plt.subplot(1,2,1)
        sns.barplot(data=counts, x=col, y="proportion")
        counts = data.groupby(["target"])[col].value_counts(normalize=True).reset_index()
        plt.subplot(1,2,2)
        sns.barplot(data=counts, x=col, y="proportion", hue="target")
        plt.suptitle(f"Chi-square {col}: p-value = {pvalue}")
        plt.show()
