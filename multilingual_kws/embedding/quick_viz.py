import glob
import os

# import tensorflow as tf
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# sns.set()
# sns.set_palette("bright")


def roc_sc(target_resuts, unknown_results):
    """
    Calculate the true positive rate (TPR) and false positive rate (FPR) for binary classification.

    Args:
        target_results (dict): Dictionary containing the results for the target class.
        unknown_results (dict): Dictionary containing the results for the unknown class.

    Returns:
        tuple: A tuple containing the TPRs, FPRs, and threshold values.
    """
    # _TARGET_ is class 1, _UNKNOWN_ is class 0

    # positive label: target keywords classified as _TARGET_
    # true positives
    target_correct = np.array(target_resuts["correct"])
    # false negatives -> target kws incorrectly classified as _UNKNOWN_:
    target_incorrect = np.array(target_resuts["incorrect"])
    total_positives = target_correct.shape[0] + target_incorrect.shape[0]

    # negative labels

    # true negatives -> unknown classified as unknown
    unknown_correct = np.array(unknown_results["correct"])
    # false positives: _UNKNOWN_ keywords incorrectly (falsely) classified as _TARGET_ (positive)
    unknown_incorrect = np.array(unknown_results["incorrect"])
    unknown_total = unknown_correct.shape[0] + unknown_incorrect.shape[0]

    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)

    tprs, fprs = [], []

    threshs = np.arange(0, 1.01, 0.01)
    for threshold in threshs:
        tpr = target_correct[target_correct > threshold].shape[0] / total_positives
        tprs.append(tpr)
        fpr = unknown_incorrect[unknown_incorrect > threshold].shape[0] / unknown_total
        fprs.append(fpr)
    return tprs, fprs, threshs


##############################################################################################
####### SINGLE LANGUAGE
##############################################################################################

# LANG_ISOCODE = "nl"

# model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/sweep_{LANG_ISOCODE}")
# if not os.path.isdir(model_dest_dir):
#     raise ValueError("no model dir", model_dest_dir)
# results_dir = model_dest_dir / "results"
# if not os.path.isdir(results_dir):
#     raise ValueError("no results dir", results_dir)

# def sc_roc_plotly(results: List[Dict]):
#     fig = go.Figure()
#     for ix, res in enumerate(results):
#         target_results = res["target_results"]
#         unknown_results = res["unknown_results"]
#         ne = res["details"]["num_epochs"]
#         nb = res["details"]["num_batches"]
#         target = res["target"]
#         curve_label = f"{target} (e:{ne},b:{nb})"
#         # curve_label=target
#         tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
#         fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=curve_label))

#     fig.update_layout(
#         xaxis_title="FPR",
#         yaxis_title="TPR",
#         title=f"{LANG_ISOCODE} 5-shot classification accuracy",
#     )
#     fig.update_xaxes(range=[0, 1])
#     fig.update_yaxes(range=[0, 1])
#     return fig


# results = []
# for pkl_file in os.listdir(model_dest_dir / "results"):
#     filename = model_dest_dir / "results" / pkl_file
#     print(filename)
#     with open(filename, "rb") as fh:
#         result = pickle.load(fh)
#         results.append(result)
# print("N words", len(results))
# fig = sc_roc_plotly(results)
# dest_plot = str(model_dest_dir / f"5shot_classification_roc_{LANG_ISOCODE}.html")
# print("saving to", dest_plot)
# fig.write_html(dest_plot)
# fig

##############################################################################################
####### MULTI LANGUAGE
##############################################################################################

# model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis/")
model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis_ooe/")
if not os.path.isdir(model_dest_dir):
    raise ValueError("no model dir", model_dest_dir)
results_dir = model_dest_dir / "results"
if not os.path.isdir(results_dir):
    raise ValueError("no results dir", results_dir)


def sc_roc_plotly(results: List[Dict]):
    """
    Generate a plotly figure for ROC (Receiver Operating Characteristic) curve.

    Args:
        results (List[Dict]): List of dictionaries containing the results for each target class.

    Returns:
        go.Figure: Plotly figure object representing the ROC curve.
    """
    fig = go.Figure()
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        ne = res["details"]["num_epochs"]
        nb = res["details"]["num_batches"]
        target_word = res["target_word"]
        target_lang = res["target_lang"]
        curve_label = f"{target_lang} {target_word} (e:{ne},b:{nb})"
        # curve_label=target
        tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
        fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=curve_label))

    fig.update_layout(
        xaxis_title="FPR",
        yaxis_title="TPR",
        title=f"5-shot classification accuracy",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


results = []
for pkl_file in os.listdir(model_dest_dir / "results"):
    filename = model_dest_dir / "results" / pkl_file
    print(filename)
    with open(filename, "rb") as fh:
        result = pickle.load(fh)
        results.append(result)
print("N words", len(results))
fig = sc_roc_plotly(results)
dest_plot = str(model_dest_dir / f"5shot_classification_roc.html")
print("saving to", dest_plot)
fig.write_html(dest_plot)
fig
