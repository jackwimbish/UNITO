import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import datashader as ds
from datashader.mpl_ext import dsshow
from scipy.spatial import ConvexHull

def clean_gate_path(s):
    # Remove "_pred" from the right end if present
    if s.endswith("_pred"):
        s = s[:-5]  # Remove last 5 characters
    
    # handle suffixes
    s = re.sub(r'_(\d+)$', r'-\1', s)
    s = re.sub(r'_neg$', r'-neg', s)
    # Remove everything prior to and including the last double underscore
    if "__" in s:
        s = s.rsplit("__", 1)[1]
    
    return s

def plot_all(gate1, gate2, x_axis, y_axis, raw_path, figure_path):
    """
    Plot gating results for all samples
    args:
        gate1: previous gates that needs to filter
        gate2: current gate
        x_axis: first gating variable
        y_axis: second gating variable
        raw_path: path containing raw data
        figure_path: path to save figures
    """

    if not os.path.exists(f"{figure_path}/Figure_{gate2}"):
        os.mkdir(f"{figure_path}/Figure_{gate2}")

    path_val = pd.read_csv(f"./Data/Data_{gate2}/pred/subj.csv")
    # find path for raw tabular data
    for subject in path_val.Image:
        plot_one(gate1, gate2, x_axis, y_axis, subject, raw_path, figure_path)

def plot_all_multi(gate_pre, gates, x_axis, y_axis, raw_path, figure_path):
    """
    Plot gating results for all samples
    args:
        gate1: previous gates that needs to filter
        gate2: current gate
        x_axis: first gating variable
        y_axis: second gating variable
        raw_path: path containing raw data
        figure_path: path to save figures
    """

    gates_str = '_'.join([clean_gate_path(g) for g in gates])
    figure_folder = f'{figure_path}/Figure_{gates_str}'
    if not os.path.exists(figure_folder):
        os.mkdir(figure_folder)

    path_val = pd.read_csv(f"./Data/Data_{gates[0]}/pred/subj.csv")
    # find path for raw tabular data
    for subject in path_val.Image:
        plot_multiple(gate_pre, gates, x_axis, y_axis, subject, raw_path, figure_path)
        
def plot_one(gate1, gate2, x_axis, y_axis, subject, raw_path, figure_path):
    """
    Plot a single sample with target gate
    args:
        gate1: previous gates that needs to filter
        gate2: current gate
        x_axis: first gating variable
        y_axis: second gating variable
        subject: sample name
        raw_path: path containing raw data
        figure_path: path to save figures
    """
    substring = f"./Data/Data_{gate2}/Raw_Numpy/"   
    subject = subject.split(substring)[1]
    substring = ".npy"
    subject = subject.split(substring)[0]
    
    data_table = pd.read_csv(f'{raw_path}/{subject}')
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ###########################################################
    # True
    ###########################################################

    data = data_table.copy()
    if gate1 != None:
        data = data[data[gate1]==1]
    
    in_gate = data[data[gate2] == 1]
    out_gate = data[data[gate2] == 0]
    in_gate = in_gate[[x_axis, y_axis]].to_numpy()
    out_gate = out_gate[[x_axis, y_axis]].to_numpy()

    if in_gate.shape[0] > 3:
        hull = ConvexHull(in_gate)
        for simplex in hull.simplices:
            ax[0].plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

        density_plot = dsshow(
            data,
            ds.Point(x_axis, y_axis),
            ds.count(),
            vmin=0,
            vmax=300,
            norm='linear',
            cmap='jet',
            aspect='auto',
            ax=ax[0],
        )

    # cbar = plt.colorbar(density_plot)
    # cbar.set_label('Number of cells in pixel')
    ax[0].set_xlabel(x_axis)
    ax[0].set_ylabel(y_axis)

    ax[0].set_title("Manual Gating")


    ###########################################################
    # predict
    ###########################################################
    
    x_axis_pred = x_axis
    y_axis_pred = y_axis

    if gate1 != None:
        gate1_pred = gate1 + '_pred'
    gate2_pred = gate2 + '_pred'

    data_table = pd.read_csv(f'./prediction/{subject}')
    data = data_table

    if gate1 != None:
        data = data[data[gate1_pred]==1]
    in_gate = data[data[gate2_pred] == 1]
    out_gate = data[data[gate2_pred] == 0]
    in_gate = in_gate[[x_axis_pred, y_axis_pred]].to_numpy()
    out_gate = out_gate[[x_axis_pred, y_axis_pred]].to_numpy()

    if in_gate.shape[0] > 3:
        hull = ConvexHull(in_gate)
        for simplex in hull.simplices:
            ax[1].plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

        density_plot = dsshow(
            data,
            ds.Point(x_axis_pred, y_axis_pred),
            ds.count(),
            vmin=0,
            vmax=300,
            norm='linear',
            cmap='jet',
            aspect='auto',
            ax=ax[1],
        )

    # cbar = plt.colorbar(density_plot)
    # cbar.set_label('Number of cells in pixel')
    ax[1].set_xlabel(x_axis)
    ax[1].set_ylabel(y_axis)

    ax[1].set_title("UNITO Auto-gating")

    # save figure
    plt.savefig(f'{figure_path}/Figure_{gate2}/Recon_Sequential_{subject}.png')
    plt.close()

def plot_multiple(pre_gate, gates, x_axis, y_axis, subject, raw_path, figure_path):
    """
    Plot a single sample with multiple target gates on the same axes
    args:
        pre_gate: previous gate that needs to filter (can be None)
        gates: list of gate names to plot
        x_axis: first gating variable
        y_axis: second gating variable
        subject: sample name
        raw_path: path containing raw data
        figure_path: path to save figures
    """
    # Process the subject name
    # Assuming consistent file structure for subject naming
    if "/" in subject:
        subject_clean = subject.split("/")[-1]
        subject_clean = subject_clean.replace(".npy", "")
    else:
        subject_clean = subject
        
    # Read the data
    data_table = pd.read_csv(f'{raw_path}/{subject_clean}')
    
    # Create a figure with two subplots (manual and predicted)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ###########################################################
    # Manual Gating (Left plot)
    ###########################################################
    data = data_table.copy()
    if pre_gate is not None:
        data = data[data[pre_gate] == 1]
    
    # Plot density first (for all cells that passed pre_gate)
    density_plot = dsshow(
        data,
        ds.Point(x_axis, y_axis),
        ds.count(),
        vmin=0,
        vmax=300,
        norm='linear',
        cmap='jet',
        aspect='auto',
        ax=ax[0],
    )
    
    # Colors for different gates
    colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    
    # Plot each gate with a different color
    for i, gate in enumerate(gates):
        color = colors[i % len(colors)]
        gate_color = color
        
        # Get cells inside this gate
        in_gate = data[data[gate] == 1]
        in_gate_numpy = in_gate[[x_axis, y_axis]].to_numpy()
        
        # Draw convex hull if enough points
        if in_gate_numpy.shape[0] > 3:
            hull = ConvexHull(in_gate_numpy)
            for simplex in hull.simplices:
                ax[0].plot(in_gate_numpy[simplex, 0], in_gate_numpy[simplex, 1], 
                         color=gate_color, linestyle='-', linewidth=2, 
                         label=clean_gate_path(gate) if simplex[0] == hull.simplices[0][0] else "")
    
    ax[0].set_xlabel(x_axis)
    ax[0].set_ylabel(y_axis)
    ax[0].set_title("Manual Gating")
    ax[0].legend(loc='best')

    ###########################################################
    # Predicted Gating (Right plot)
    ###########################################################
    
    # Load prediction data
    data_table_pred = pd.read_csv(f'./prediction/{subject_clean}')
    data_pred = data_table_pred
    
    pre_gate_pred = pre_gate + '_pred' if pre_gate is not None else None
    
    if pre_gate is not None:
        data_pred = data_pred[data_pred[pre_gate_pred] == 1]
    
    # Plot density for prediction (for all cells that passed pre_gate)
    density_plot_pred = dsshow(
        data_pred,
        ds.Point(x_axis, y_axis),
        ds.count(),
        vmin=0,
        vmax=300,
        norm='linear',
        cmap='jet',
        aspect='auto',
        ax=ax[1],
    )
    
    # Plot each predicted gate with a different color
    for i, gate in enumerate(gates):
        color = colors[i % len(colors)]
        gate_color = color
        gate_pred = gate + '_pred'
        
        # Get cells inside this gate
        in_gate_pred = data_pred[data_pred[gate_pred] == 1]
        in_gate_pred_numpy = in_gate_pred[[x_axis, y_axis]].to_numpy()

        used_label_names = set()
        clean_gate = clean_gate_path(gate)
        
        # Draw convex hull if enough points
        if in_gate_pred_numpy.shape[0] > 3:
            hull_pred = ConvexHull(in_gate_pred_numpy)
            for simplex in hull_pred.simplices:
                ax[1].plot(in_gate_pred_numpy[simplex, 0], in_gate_pred_numpy[simplex, 1], 
                         color=gate_color, linestyle='-', linewidth=2,
                         label=clean_gate if simplex[0] == hull_pred.simplices[0][0] and clean_gate not in used_label_names else "")
                used_label_names.add(clean_gate)
    
    ax[1].set_xlabel(x_axis)
    ax[1].set_ylabel(y_axis)
    ax[1].set_title("UNITO Auto-gating")
    ax[1].legend(loc='best')
    
    plt.tight_layout()
    
    # Generate a combined name for the figure
    gates_str = "_".join([clean_gate_path(g) for g in gates])
    figure_folder = f'{figure_path}/Figure_{gates_str}'
    figure_filename = f'{figure_folder}/Recon_Sequential_{gates_str}_{subject_clean}.png'
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(figure_filename), exist_ok=True)
    
    plt.savefig(figure_filename)
    plt.close()
    
    return fig
