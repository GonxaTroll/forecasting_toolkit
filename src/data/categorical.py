import pandas as pd
def agg_categories(data: pd.DataFrame, column: str, n_top_categories: int=4,
                   train_dataset: pd.DataFrame = None):
    if train_dataset is not None:
        top_categories = train_dataset[column][train_dataset[column]!="OTHER"]
    else:
        top_categories = data[column].value_counts()[:n_top_categories].index.tolist()
    def agg_categories_aux(value, top_categories):
        if value is not None and value not in top_categories: ############################3 how to check for null values
            return "OTHER"
        return value
    return data[column].map(lambda x: agg_categories_aux(x, top_categories))