"""
Take mock data and try to have it live-plotted as it increases in
size after each trial.

author Tim Maniquet
created 4 November 2024
"""

# Global imports
import dash
import warnings
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import os, glob
import json

# Local imports
from src import palette
from src.utils import *
from plotting.plot_utils import *

## PARAMETERS

# App detail
UPDATE_FREQUENCY = 500 # in ms
APP_TITLE = "Live mouse tracker results"
N_FIGURES = 4 # how many plots are showing

# Directories
DATA_DIR = './data'
DATAFILE_EXT = '.json'

# Start a dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1(APP_TITLE),
    
    # Create multiple plot components
    # html.Div([
    #     html.Div(dcc.Graph(id=f'live-update-plot-{i}'), className="plot-container")
    #     for i in range(1, N_FIGURES + 1)  # For 5 plots
    # ]),
    # Create a 2x2 grid for the plots
    html.Div([
        html.Div(dcc.Graph(id=f'live-update-plot-{i}'), className="plot-container") for i in range(1, 5)
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(2, 1fr)',  # Two columns
        'gridGap': '20px',  # Space between the plots
        'maxWidth': '1200px',  # Set a max width for the whole grid
        'margin': '0 auto'  # Center the grid horizontally
    }),
    
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_FREQUENCY,  # Update every half-second
        n_intervals=0
    )
])

# Callback to update the graph every second
@app.callback(
    [Output(f'live-update-plot-{i}', 'figure') for i in range(1, N_FIGURES + 1)],
    Input('interval-component', 'n_intervals')
)
def update_graph_live(n):
    
    # List all the corresponding data files 
    data_files = glob.glob(os.path.join(DATA_DIR, f'*{DATAFILE_EXT}'))
    if len(data_files) == 0:
        # Return empty plots if there's no data
        return [go.Figure() for _ in range(5)]
    else:
        # Find the most recent one
        data_file = get_latest_file(data_files)

    # Load data from the latest trial file
    with open(data_file, 'r') as file:
        data = json.load(file)
    
    # Unpack the mouse tracking data
    unpacked_data = [extend_fill_trial(trial) for trial in data]
        
    # Fetch the window size from the data
    win_size = (1200, 800)
    conditions = list(palette.keys())
        
    # Extract the condition-wise trajectories
    condition_trajectories, average_trajectories = unpack_cond_trajectories(unpacked_data)
    # Extract the image-wise trajectories
    _, avg_img_trajectories = unpack_img_trajectories(unpacked_data)
    # extract image names from the output
    img_names = list(avg_img_trajectories.keys())
    # extract the time-resolved difference matrices & re-order them
    matrices = calculate_pairwise_diff_matrices(avg_img_trajectories)
    matrices, labels = reorder_matrices_by_conditions(matrices, img_names, conditions)
    
    # Generate each plot using different external functions
    figures = [
        plot_trajectory_last_trial(data, win_size, palette),
        plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette),
        plot_matrices(matrices, [100, 400, 700, 900]),
        plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette),
    ]
    
    # Assert that we have all the figures we need
    if len(figures) != N_FIGURES:
        warnings.warn(f"Warning: {len(figures)} figures generated (expected {N_FIGURES}).")
    
    return figures

if __name__ == '__main__':
    app.run_server(debug=True)