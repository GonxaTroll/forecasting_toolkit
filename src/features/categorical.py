import pandas as pd
def categorical_to_dummies(data: pd.DataFrame, dtype=8):
    if dtype is None:
        dtype = 64
    categorical_cols = []
    for col in data.columns:
        if data[col].dtype == "object":
            if data[col].isnull().sum() == 0:
                drop_first = True
            else:
                drop_first = False
            data = pd.concat([data, pd.get_dummies(data[col], dtype=f"int{dtype}",
                                                   drop_first=drop_first, prefix=f"{col}_")],
                            axis=1)
            categorical_cols.append(col)
    data = data.drop(columns=categorical_cols)
    return data

