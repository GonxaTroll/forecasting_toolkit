import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def plot_null_percent(data: pd.DataFrame, proportion: float = 0.7):
    plt.figure(figsize=(35,5))
    nulls = (data.isnull().sum() / len(data)).sort_values(ascending=False)
    nulls = pd.DataFrame(nulls).reset_index()
    nulls.columns = ["variable", "null_percentage"]
    sns.barplot(data=nulls, x="variable", y="null_percentage")
    plt.xticks(rotation=45)
    # plt.hlines(y=[proportion], linestyles=["-"], xmin=0, xmax=1)
    # fig.refline(y=proportion)
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