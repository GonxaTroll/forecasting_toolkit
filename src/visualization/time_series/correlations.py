"""correlations.py
This script contains functions to calculate and plot correlations between time series.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
# from scipy.stats import spearmanr

def calculate_correlations(data: pd.DataFrame, variable1: str, variable2: str, start: int = 0,
                           end: int = 12,
                           search_best_in_window_width: int = 12,
                           correlation_method: str = "pearson"):
    if variable1 == variable2:
        data = data.loc[:,[variable1]]
    else:
        data = data.loc[:,[variable1,variable2]]
    data = data.dropna()
    max_shift = len(data.index.unique())-2
    best_correlation = [0,0] #shift_value, correlation_value
    if start == "max":
        start = (-1)*max_shift
    if end == "max":
        end = max_shift
    corr_values = []
    shift_values = list(range(start,end+1))
    for shift in shift_values:
        if correlation_method == "pearson":
            corr = data[variable1].corr(data[variable2].shift(shift))
        elif correlation_method == "spearman":
            corr = data[variable1].corr(data[variable2].shift(shift), method="spearman")
        corr_values.append(corr)
        if abs(shift) <= search_best_in_window_width and abs(corr) > abs(best_correlation[1]):
            best_correlation[0], best_correlation[1] = shift, corr
    return shift_values,corr_values


def plot_correlations(data, variable1, variable2, start=0, end=24):
    if isinstance(variable2, list):
        title = f"Correlations between {variable1} and variables"
        fig = go.Figure()
        for var in variable2:
            shift_values, corr_values = calculate_correlations(data,variable1,var, start=start,end=end)
            fig.add_scatter(x=shift_values, y=corr_values, name=var, mode="lines+markers")
            maxed, shift = max([(abs(c),s) for c,s in zip(corr_values,shift_values)])
            # fig.add_scatter(x=[shift],y=[maxed],marker_size=8,marker_color="red",name="Max")
            fig.add_vline(x=shift,line_dash="dash",line_color="red",opacity=0.7,annotation_text="Max")
    else:
        title = f"Correlaciones entre variables y {variable2}"
        if isinstance(variable1, str):
            variable1 = [variable1]
        fig = go.Figure()
        for var in variable1:
            shift_values, corr_values = calculate_correlations(data,var,variable2, start=start,end=end)
            fig.add_scatter(x=shift_values, y=corr_values, name=var, mode="lines+markers")
            maxed,shift = max([(abs(c),s) for c,s in zip(corr_values,shift_values)])
            # fig.add_scatter(x=[shift],y=[maxed],marker_size=8,marker_color="red",name="Max")
            fig.add_vline(x=shift,line_dash="dash",line_color="red",opacity=0.7,annotation_text="Max")

    fig.update_layout(title= dict(text=title,font_size=20,x=0.5),
                      xaxis= dict(title="Shift",titlefont_size=20),
                      yaxis= dict(title="Correlación",titlefont_size=20,range=[-1,1]))
    return fig
