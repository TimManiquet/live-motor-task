"""
Take mock data and try to have it live-plotted as it increases in
size after each trial.

author Tim Maniquet
created 4 November 2024
"""

# Global imports
import dash_bootstrap_components as dbc
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

#########
import sys
sys.path.append(".")
#########

# Local imports
from src import palettes
from src.utils import *
from plot_utils.plot_utils import *

## PARAMETERS

# App detail
FAST_UPDATE_FREQUENCY = 2000 # in ms
SLOW_UPDATE_FREQUENCY = 10000 # in ms
APP_TITLE = "Live mouse tracker results"
# N_FIGURES = 4 # how many plots are showing

# Directories
DATA_DIR = 'data'
DATAFILE_EXT = '.json'

# Declare the time points to show matrices at
time_points = [300, 400, 500, 600, 700, 800]

# Global cache to store data and track processed lines in each file
cached_data = []
file_line_tracker = {}  # Tracks the last processed line for each file

# Read the external ratings (brain, gpt, ...)
ratings_file = os.path.join('src', 'external', 'ratings.tsv')
ratings = pd.read_table(ratings_file)

# For the correlations to show: Declare the different targets to display
targets = ['gpt', 'low', 'mid', 'high']
expanded_target_names = ['chatGPT 4.0', 'low-level cortical areas', 'mid-level cortical areas', 'high-level cortical areas']

# Declare the three types of valences
valences = ['people', 'scene', 'image']

# Calculate the number of graphs that makes
n_corr_plots = len(targets) * len(valences)

## DATA HANDLING FUNCTIONS

def load_all_data():
    """Initial load of all JSON files and cache their content."""
    global cached_data, file_line_tracker
    cached_data = []
    file_line_tracker = {}

    json_files = glob.glob(os.path.join(DATA_DIR, f'*{DATAFILE_EXT}'))
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            preprocessed_data = [extend_fill_trial(trial) for trial in data]
            cached_data.extend(preprocessed_data)
            file_line_tracker[file] = len(data)  # Track how many lines we've processed

def update_cached_data():
    """Update cached data by checking for new files or new lines in the last file."""
    global cached_data, file_line_tracker
    json_files = glob.glob(os.path.join(DATA_DIR, f'*{DATAFILE_EXT}'))

    # Check for new or updated files
    for file in json_files:
        if file not in file_line_tracker:
            # New file detected, load it completely
            with open(file, 'r') as f:
                data = json.load(f)
                preprocessed_data = [extend_fill_trial(trial) for trial in data]
                cached_data.extend(preprocessed_data)
                file_line_tracker[file] = len(preprocessed_data)
        else:
            # Check for new lines in the existing file
            with open(file, 'r') as f:
                data = json.load(f)
                new_lines = data[file_line_tracker[file]:]  # Get only the new lines
                preprocessed_newlines = [extend_fill_trial(trial) for trial in new_lines]
                cached_data.extend(preprocessed_newlines)
                file_line_tracker[file] = len(data)

# Initial load of data
load_all_data()

# Ignore the warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Start a dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = html.Div([
    
    # Main app title
    html.H1(APP_TITLE),
    
    # Dropdowns for selecting data, aligned horizontally
    html.Div([
        # First dropdown
        html.Div([
            html.Div("Choose data to plot:", style={"font-size": "18px", "margin-right": "8px"}),
            dbc.Select(
                id='data-option',
                options=[
                    {'label': 'All data', 'value': 'all_data'},
                    {'label': 'Last participant', 'value': 'last_participant'}
                ],
                value='all_data',
                style={"width": "25%"}
            )
        ], style={"width": "45%"}),  # Left dropdown takes 45% of the available width

        # Second dropdown
        html.Div([
            html.Div("Choose colours to display:", style={"font-size": "18px", "margin-left": "8px"}),
            dbc.Select(
                id='palette-option',
                options=[
                    {'label': 'People valence', 'value': 'people_valence'},
                    {'label': 'Context', 'value': 'context'},
                    {'label': 'People interactions', 'value': 'interaction'},
                    {'label': 'Congruency people-context', 'value': 'congruency'}
                ],
                value='people_valence',
                style={"width": "25%"}
            )
        ], style={"width": "50%"})  # Second dropdown takes the other half
    ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "20px"}),

    # Top two graphs: mouse trajectories
    html.Div([
        html.H2("Mouse trajectories"),
        dcc.Graph(id='fast-graph-1'),
        dcc.Graph(id='fast-graph-2'),
        dcc.Interval(
            id='fast-interval',
            interval=FAST_UPDATE_FREQUENCY,  # Update every 5 seconds
            n_intervals=0
        )
    ], style={'margin-bottom': '50px'}),

    # # Bottom graphs
    # html.Div([
    #     html.H2("Correlations"),
    #     dcc.Graph(id='slow-graph-1'),
    #     dcc.Graph(id='slow-graph-2'),
    #     dcc.Interval(
    #         id='slow-interval',
    #         interval=SLOW_UPDATE_FREQUENCY,  # Update every 60 seconds
    #         n_intervals=0
    #     )
    # ]),
    
    # Tabs for the slow graphs
    dcc.Tabs(id='tabs-container', children=[
        dcc.Tab(label=f"{targets[i]}", children=[
            html.Div([
                html.H3(f"Correlations with {expanded_target_names[i]}"),
                dcc.Graph(id=f'slow-graph-{i*3+1}'),
                dcc.Graph(id=f'slow-graph-{i*3+2}'),
                dcc.Graph(id=f'slow-graph-{i*3+3}')
            ], style={'margin-bottom': '20px'})
        ]) for i in range(4)  # Create 4 tabs, each with 3 graphs
    ]),

    # Slow interval for updating all slow graphs
    dcc.Interval(
        id='slow-interval',
        interval=SLOW_UPDATE_FREQUENCY,
        n_intervals=0
    )
    # # Grid layout for the plots
    # html.Div([
    #     # Plots 1-4
    #     html.Div(dcc.Graph(id=f'live-update-plot-{i}'), className="plot-container") for i in range(1, N_FIGURES)
    # ] + [
    #     # Legend plot
    #     html.Div(dcc.Graph(id='legend'), className="plot-container")
    # ], style={
    #     'display': 'grid',
    #     'gridTemplateColumns': 'repeat(2, 1fr)',  # Three columns
    #     'gridGap': '20px',
    #     'maxWidth': '1600px',
    #     'margin': '0 auto'
    # }),

    # # Interval for live updates
    # dcc.Interval(
    #     id='interval-component',
    #     interval=UPDATE_FREQUENCY,
    #     n_intervals=0
    # )
])

# Callback to update the top two (fast) graphs
@app.callback(
    # [Output(f'live-update-plot-{i}', 'figure') for i in range(1, N_FIGURES)] +
    # [Output('legend', 'figure')],
    [Output('fast-graph-1', 'figure'), Output('fast-graph-2', 'figure')],
    [Input('fast-interval', 'n_intervals'),
     Input('data-option', 'value'),
     Input('palette-option', 'value'),]
)
def update_fast_graphs(n, data_option, palette_option):
    
    # Update the cached data
    update_cached_data()  # Update cache with new files or lines
        
    # Fetch the window size (preferably from the data)
    win_size = (1200, 800)
    # Extract the palette based on the user's choice
    palette = palettes[palette_option]
        
    # If we should only plot the last participant
    if data_option ==  'last_participant':
        # List all the files
        data_files = glob.glob(os.path.join(DATA_DIR, f'*{DATAFILE_EXT}'))
        # Fetch the last one
        last_file = get_latest_file(data_files)
        # Check how many lines to take from it
        n_lines = file_line_tracker[last_file]
        # Upate the data accordinly
        # cached_data = cached_data[-n_lines:]
        # Extract the condition-wise trajectories
        condition_trajectories, average_trajectories = unpack_cond_trajectories(cached_data[-n_lines:], palette)
    elif data_option == 'all_data':
        condition_trajectories, average_trajectories = unpack_cond_trajectories(cached_data, palette)
    
    # Extract the image-wise trajectories
    # _, avg_img_trajectories = unpack_img_trajectories(cached_data)
    
    # Extract image names from the output
    # img_names = list(avg_img_trajectories.keys())
    
    # Extract the time-resolved difference matrices & re-order them
    # matrices = calculate_pairwise_diff_matrices(avg_img_trajectories)
    # reordered_matrices, labels = reorder_matrices_by_conditions(matrices, img_names, conditions)
    # Extract dimensionality reduction coordinates from the matrices
    # dim_reduc_coordinates = dim_reduc_matrices(matrices, time_points, dim_reduc_algorithm)
    
    # Generate each plot using different external functions
    # figures = []
    # figures.append(plot_trajectory_last_trial(cached_data[-1], win_size, palette))
    # figures.append(plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette))
    fig1 = plot_trajectory_last_trial(cached_data[-1], win_size, palette)
    fig2 = plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette)

    # Assert that we have all the figures we need
    # if len(figures) != N_FIGURES:
    #     warnings.warn(f"Warning: {len(figures)} figures generated (expected {N_FIGURES}).")
    
    return fig1, fig2

# Callback for the bottom (slow) graphs
@app.callback(
    [Output(f'slow-graph-{i}', 'figure') for i in range(1, n_corr_plots+1)],
    [Input('slow-interval', 'n_intervals'),
     Input('data-option', 'value'),
     Input('palette-option', 'value'),]
)
def update_slow_graphs(n, data_option, palette_option):
    
    # Update the cached data
    update_cached_data()  # Update cache with new files or lines
    
    # Extract the palette based on the user's choice
    palette = palettes[palette_option]
        
    # If we should only plot the last participant
    if data_option ==  'last_participant':
        # List all the files
        data_files = glob.glob(os.path.join(DATA_DIR, f'*{DATAFILE_EXT}'))
        # Fetch the last one
        last_file = get_latest_file(data_files)
        # Check how many lines to take from it
        n_lines = file_line_tracker[last_file]
        # Upate the data accordinly
        # cached_data = cached_data[-n_lines:]
        # Extract the condition-wise trajectories
        condition_trajectories, average_trajectories = unpack_cond_trajectories(cached_data[-n_lines:], palette)
    elif data_option == 'all_data':
        condition_trajectories, average_trajectories = unpack_cond_trajectories(cached_data, palette)
    
    # Extract the image-wise trajectories
    _, avg_img_trajectories = unpack_img_trajectories(cached_data)
    
    # Start calculating correlations
    correlations = {
        f'{valence_target}-valence-{corr_target}': []
        for valence_target in valences
        for corr_target in targets
    }
    
    # Fetch the images we have so far
    images = list(avg_img_trajectories.keys())
    
    # Extract the time range from a random image
    # time_range = len(avg_img_trajectories[images[0]]['x_coord'])
    
    # Pre-extract x-coordinates as a NumPy array (n_images × time_range)
    x_coords = np.array([avg_img_trajectories[img]['x_coord'] for img in images])  # shape: (n_images, time_range)

    # Pre-extract ratings for all correlation targets into dictionaries of arrays
    ratings_dict = {
        c: np.array([ratings.loc[ratings['filename'] == img, c].values[0] for img in images])
        for c in correlations.keys()
    }

    # Initialize correlations with nan arrays (for vectorized assignment)
    time_range = x_coords.shape[1]
    correlations = {c: np.full(time_range, np.nan) for c in correlations.keys()}

    # Go over each time point in the time range
    for t in range(time_range):
        x_at_t = x_coords[:, t]  # Extract all x-coordinates at time t

        if not np.any(np.isnan(x_at_t)):  # Check for nan values only once
            for c, target in ratings_dict.items():
                correlations[c][t] = np.corrcoef(x_at_t, target)[0, 1]  # Compute correlation
    
    # Make a plot from the correlations for each valence type and correlation target
    figs = [plot_time_correlation(correlations[target], time_range) for target in correlations.keys()]
    # fig1.show()
    # fig2 = plot_time_correlation(correlations[target2], time_range)

    return figs

# for img in avg_img_trajectories:
#     print(
#         img,
#         np.nanmean(avg_img_trajectories[img]['x_coord'])
#     )
#     trace = np.nanmean(avg_img_trajectories[img]['x_coord'])
#     plt.plot(trace)
#     plt.show()

# plt.plot(correlations[c])
# plt.show()

if __name__ == '__main__':
    # Define host and port
    host = '127.0.0.1'  # Default host
    port = 8050          # Default port

    # Construct the URL
    url = f"http://{host}:{port}"

    # Copy the URL to the clipboard
    # pyperclip.copy(url)
    # print(f"URL copied to clipboard: {url}")
    
    # Run the app
    # app.run_server(debug=True)
    app.run_server(host=host, port=port, debug=True)