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
# from sklearn.manifold import TSNE, MDS
# from sklearn.decomposition import PCA
import warnings # to ignore warnings
from sklearn.decomposition import IncrementalPCA
import pyperclip # to copy the link to the app

#########
import sys
sys.path.append(".")
#########

# Local imports
from src import palette
from src.utils import *
from plot_utils.plot_utils import *

## PARAMETERS

# App detail
UPDATE_FREQUENCY = 1000 # in ms
APP_TITLE = "Live mouse tracker results"
N_FIGURES = 4 # how many plots are showing

# Directories
DATA_DIR = './data'
DATAFILE_EXT = '.json'

# Declare the time points to show matrices at
time_points = [300, 400, 500, 600, 700, 800]

# Dimensionality reduction technique
# dim_reduc_algorithm = TSNE(n_components=2, random_state=42)
# dim_reduc_algorithm = PCA(n_components=2, random_state=42)
dim_reduc_algorithm = IncrementalPCA(n_components=2)
# dim_reduc_algorithm = MDS(n_components=2,dissimilarity='euclidean', random_state=42)

# Ignore the warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Start a dash app
app = dash.Dash(__name__)

# Define the layout of the app
# app.layout = html.Div([
#     html.H1(APP_TITLE),
    
#     # Create multiple plot components
#     # html.Div([
#     #     html.Div(dcc.Graph(id=f'live-update-plot-{i}'), className="plot-container")
#     #     for i in range(1, N_FIGURES + 1)  # For 5 plots
#     # ]),
#     # Create a 2x2 grid for the plots
#     html.Div([
#         html.Div(dcc.Graph(id=f'live-update-plot-{i}'), className="plot-container") for i in range(1, 5)
#     ], style={
#         'display': 'grid',
#         'gridTemplateColumns': 'repeat(2, 1fr)',  # Two columns
#         'gridGap': '20px',  # Space between the plots
#         'maxWidth': '1200px',  # Set a max width for the whole grid
#         'margin': '0 auto'  # Center the grid horizontally
#     }),
    
#     dcc.Interval(
#         id='interval-component',
#         interval=UPDATE_FREQUENCY,  # Update every half-second
#         n_intervals=0
#     )
# ])

# # Callback to update the graph every second
# @app.callback(
#     [Output(f'live-update-plot-{i}', 'figure') for i in range(1, N_FIGURES + 1)],
#     Input('interval-component', 'n_intervals')
# )

# Define the layout of the app
app.layout = html.Div([
    html.H1(APP_TITLE),
    
    # Grid layout for the plots
    html.Div([
        # Plots 1-4
        html.Div(dcc.Graph(id=f'live-update-plot-{i}'), className="plot-container") for i in range(1, N_FIGURES)
    ] + [
        # Legend plot
        html.Div(dcc.Graph(id='legend'), className="plot-container")
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(2, 1fr)',  # Three columns
        'gridGap': '20px',
        'maxWidth': '1600px',
        'margin': '0 auto'
    }),

    # Interval for live updates
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_FREQUENCY,
        n_intervals=0
    )
])

# Callback to update the graphs
@app.callback(
    [Output(f'live-update-plot-{i}', 'figure') for i in range(1, N_FIGURES)] +
    [Output('legend', 'figure')],
    Input('interval-component', 'n_intervals')
)

def update_graph_live(n):
    
    # List all the corresponding data files 
    data_files = glob.glob(os.path.join(DATA_DIR, f'*{DATAFILE_EXT}'))
    if len(data_files) == 0:
        # Return empty plots if there's no data
        return [go.Figure() for _ in range(N_FIGURES)]
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
    # conditions = list(palette.keys())
    conditions = ['positive', 'ambiguous', 'negative'] # declare them explicity for a nice order
        
    # Extract the condition-wise trajectories
    condition_trajectories, average_trajectories = unpack_cond_trajectories(unpacked_data)
    
    # Extract the image-wise trajectories
    _, avg_img_trajectories = unpack_img_trajectories(unpacked_data)
    
    # Extract image names from the output
    img_names = list(avg_img_trajectories.keys())
    
    # Extract the time-resolved difference matrices & re-order them
    matrices = calculate_pairwise_diff_matrices(avg_img_trajectories)
    reordered_matrices, labels = reorder_matrices_by_conditions(matrices, img_names, conditions)
    # Extract dimensionality reduction coordinates from the matrices
    dim_reduc_coordinates = dim_reduc_matrices(matrices, time_points, dim_reduc_algorithm)
    
    # Generate each plot using different external functions
    figures = []
    figures.append(plot_trajectory_last_trial(data, win_size, palette))
    figures.append(plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette))
    figures.append(plot_matrices_and_dim_reduc(matrices, time_points, dim_reduc_algorithm, labels, palette))
    figures.append(custom_legend(palette))
    # figures.append(plot_matrices(reordered_matrices, time_points))
    # figures.append(plot_dim_reduc(dim_reduc_coordinates, time_points, labels, palette))
    
    # figures = [
    #     plot_trajectory_last_trial(data, win_size, palette),
    #     plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette),
    #     plot_matrices(reordered_matrices, time_points),
    #     # plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette),
    #     plot_dim_reduc(dim_reduc_coordinates, time_points, labels, palette)
    # ]

    # Assert that we have all the figures we need
    if len(figures) != N_FIGURES:
        warnings.warn(f"Warning: {len(figures)} figures generated (expected {N_FIGURES}).")
    
    return figures

if __name__ == '__main__':
    # Define host and port
    host = '127.0.0.1'  # Default host
    port = 8050          # Default port

    # Construct the URL
    url = f"http://{host}:{port}"

    # Copy the URL to the clipboard
    pyperclip.copy(url)
    print(f"URL copied to clipboard: {url}")
    
    # Run the app
    # app.run_server(debug=True)
    app.run_server(host=host, port=port, debug=True)