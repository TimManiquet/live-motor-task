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
import pandas as pd
import os, glob
import json
import warnings # to ignore warnings$
import base64 # to encode images and display them

#########
import sys
sys.path.append(".")
#########

# Local imports
from src import palettes, target_colours, ratings_palette
from src.utils import *
from plot_utils.plot_utils import *

## PARAMETERS

# App detail
FAST_UPDATE_FREQUENCY = 2000 # in ms
UPDATE_FREQUENCY = 10000 # in ms
SLOW_UPDATE_FREQUENCY = 30000 # in ms
APP_TITLE = "Live mouse tracker results"

# Directories
DATA_DIR = 'data'
DATAFILE_EXT = '.json'

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

# Calculate the number of slow graphs to make from targets and valences
n_corr_plots = len(targets) * len(valences)

# Make a table to show image ratings
images = ratings['filename']

# Function to determine cell color based on rating
def get_color(value, rgb):
    """Returns a linear color for values between -1 and 8.
    The linear scale is dependent on the rgb values given as input, with 
    the rgb value as the lowest and white as the highest value"""
    # Normalize the value to a range of 0 to 1
    norm_value = (value + 1) / 9  # Scale value to be between 0 and 1
    norm_value = max(0, min(1, norm_value))  # Clamp the value within [0, 1]

    # Calculate the RGB components for a linear gradient from dodgerblue to white
    r = int(rgb[0] + (255 - rgb[0]) * norm_value)   # Red goes from 30 to 255
    g = int(rgb[1] + (255 - rgb[1]) * norm_value) # Green goes from 144 to 255
    b = int(rgb[2] + (255 - rgb[2]) * norm_value) # Blue stays at 255 (full blue)

    return f"rgb({r}, {g}, {b})"


# Build the table rows
table_rows = []
# for img, rating in zip(images, ratings):
for _, row in ratings.iterrows():
    # Extract the image filename
    # img = os.path.join('assets', 'stimuli', row['filename'])
    filename = os.path.join('stimuli', row['filename'])
    encoded_image = base64.b64encode(open(filename, 'rb').read())
    
    # Make a list of cells to construct the table
    list_of_cells = []
    for target in targets:
        for valence in valences:
            cell = html.Td(
                f"{row[f'{valence}-valence-{target}']:.2f}",
                style={"background-color": get_color(row[f'{valence}-valence-{target}'], target_colours[target]), "text-align": "center"}
            )
            # Add the result to the list
            list_of_cells.append(cell)

    # Construct the row
    table_row = html.Tr([
        # Image cell
        html.Td(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={"width": "140px", "height": "100px"})),
        # Rating cells
        *list_of_cells
    ])
    
    table_rows.append(table_row)


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
        ], style={"flex": "1"}),  # Left dropdown takes 45% of the available width

        # Second dropdown
        html.Div([
            html.Div("Choose colours to display:", style={"font-size": "18px", "margin-left": "8px"}),
            dbc.Select(
                id='palette-option',
                options=[
                    {'label': 'People valence', 'value': 'people_valence'},
                    {'label': 'Context', 'value': 'context'},
                    {'label': 'People interactions', 'value': 'interaction'},
                    {'label': 'Congruency people-context', 'value': 'congruency'},
                    {'label': 'Congruency people-context & people valence', 'value': 'congruency_and_people'}
                ],
                value='people_valence',
                style={"width": "40%"}
            )
        ], style={"flex": "1"})  # Second dropdown takes the other half
    ], style={
        "display": "flex",
        "flex-wrap": "wrap",  # Allow wrapping on smaller screens
        "justify-content": "space-between", 
        "margin-bottom": "20px"
    }),

    # Top two graphs: mouse trajectories
    html.Div([
        html.H2("Mouse trajectories"),
        html.Div([
            dcc.Graph(id='fast-graph-1', style={"flex": "1", "height": "700px"}),
            dcc.Graph(id='fast-graph-2', style={"flex": "1", "height": "700px"})
        ], style={
            "display": "flex",
            "gap": "20px",  # Space between the two graphs
            "flex-wrap": "wrap",  # Allow wrapping on smaller screens
            "margin-bottom": "20px"
        }),
        dcc.Interval(
            id='fast-interval',
            interval=FAST_UPDATE_FREQUENCY,  # Update every 5 seconds
            n_intervals=0
        )
    ], style={'margin-bottom': '50px'}),

    # Tabs for the middle correlation graphs
    dcc.Tabs(id='tabs-container', children=[
        dcc.Tab(label=f"{targets[i]}", children=[
            # One div per target
            html.Div([
                # Title of the correlation tab
                html.H3(f"Correlations with {expanded_target_names[i]}"),
                # Message for the drop-down selection
                html.Div("Choose images to correlate:", style={"font-size": "18px", "margin-left": "8px"}),
                # Selection module to choose images to correlate
                dbc.Select(
                    id='congruency-option',
                    options=[
                        {'label': 'Congruent images', 'value': 'congruent'},
                        {'label': 'Incongruent images', 'value': 'incongruent'},
                        {'label': 'All images', 'value': 'all'}
                    ],
                    value='all',
                    style={"width": "40%"}
                ),
                # Correlation graph itself
                dcc.Graph(id=f'middle-graph-{i}')
            ], style={'margin-bottom': '20px'})
        ]) for i in range(4)  # Create 4 tabs, each with 3 graphs
    ]),

    # Normal interval for updating all middle graphs
    dcc.Interval(
        id='interval',
        interval=UPDATE_FREQUENCY,
        n_intervals=0
    ),

    # Leave some space here
    html.Br(), html.Br(), html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
    
    # Ratings table
    html.H2("Image Ratings Table"),
    html.Table(
        # Table header
        [html.Tr([
            html.Th("Image"),
            *[html.Th(f"{target} {valence}")
              for target in targets
              for valence in valences]
            # html.Th("Source 1"), html.Th("Source 2"), html.Th("Source 3"), html.Th("Source 4")
        ])] + table_rows,  # Combine header and rows
        style={"width": "100%", "border-collapse": "collapse", "border": "1px solid black"}
    ),
    
    # Leave some space here
    html.Br(), html.Br(), html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
    
    # Very bottom: the slow, detailed graphs
    html.H2("Detailled correlations"),
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
])

# Callback to update the top two (fast) graphs
@app.callback(
    [Output('fast-graph-1', 'figure'), Output('fast-graph-2', 'figure')],
    [Input('fast-interval', 'n_intervals'),
     Input('data-option', 'value'),
     Input('palette-option', 'value')]
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
        # Extract the condition-wise trajectories
        condition_trajectories, average_trajectories = unpack_cond_trajectories(cached_data[-n_lines:], palette)
    elif data_option == 'all_data':
        condition_trajectories, average_trajectories = unpack_cond_trajectories(cached_data, palette)
    
    # Make the two mouse trajectory plots
    fig1 = plot_trajectory_last_trial(cached_data[-1], win_size, palette)
    fig2 = plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette)
    
    return fig1, fig2


# Callback for the middle correlation graphs
@app.callback(
    [Output(f'middle-graph-{i}', 'figure') for i in range(len(targets))],
    [Input('interval', 'n_intervals'),
     Input('congruency-option', 'value')]
)
def update_middle_graphs(n, congruency_option):
    
    # Update the cached data
    update_cached_data()  # Update cache with new files or lines
        
    # Extract the image-wise trajectories
    _, avg_img_trajectories = unpack_img_trajectories(cached_data)
    
    # Initiate correlations with people, scene, and image valence across targets
    correlations = {
        f'{corr_target}': {f'{valence_target}-valence': []
        for valence_target in valences}
        for corr_target in targets
    }
    
    # Fetch the images based on the congruency option
    if congruency_option == 'all':
        # take all images
        images = list(avg_img_trajectories.keys())
    elif congruency_option == 'congruent':
        # take only congruent images
        images = [img for img in list(avg_img_trajectories.keys()) if 'congruent' in img]
    elif congruency_option == 'incongruent':
        # take only incongruent images
        images = [img for img in list(avg_img_trajectories.keys()) if 'incongruent' in img]
    
    # Fetch the indices of the images we have
    # ratings_palette_idx = {condition: [i for i, img in enumerate(images) if condition in img]
    #     for condition in ratings_palette.keys()}
    
    # Pre-extract x-coordinates as a NumPy array (n_images × time_range)
    x_coords = np.array([avg_img_trajectories[img]['x_coord'] for img in images])  # shape: (n_images, time_range)

    # Pre-extract ratings for all correlation targets into dictionaries of arrays
    ratings_dict = {
        target: {valence: np.array([ratings.loc[ratings['filename'] == img, f'{valence}-valence-{target}'].values[0] for img in images])
        for valence in valences}
        for target in targets
    }

    # Initialize correlations with nan arrays (for vectorized assignment)
    time_range = x_coords.shape[1]
    # correlations = {c: {p: np.full(time_range, np.nan)}
    correlations = {
        target: {f'{valence}_valence': np.full(time_range, np.nan)
        for valence in valences}
        for target in targets
    }

    # Loop over each target to correlate
    for target in targets:
        # Loop over each valence to correlate
        for valence in valences:
            # Extract the ratings for the given target and valence
            target_valence_ratings = ratings_dict[target][valence]
            # Loop over time points and report a correlation
            for t in range(time_range):
                x_at_t = x_coords[:, t]  # Extract all x-coordinates at time t
                if not np.any(np.isnan(x_at_t)):  # Check for nan values only once
                    correlations[target][f'{valence}_valence'][t] = np.corrcoef(x_at_t, target_valence_ratings)[0, 1]  # Compute correlation
    
    # Make a plot from the correlations for each valence type and correlation target
    figs = []
    for target in targets:
        figs.append(plot_time_correlations(correlations[target], f"", ratings_palette))
    
    return figs

### WORKING ZONE HERE
# Callback for the middle graphs correlation graphs
@app.callback(
    [Output(f'slow-graph-{i}', 'figure') for i in range(1, n_corr_plots+1)],
    [Input('slow-interval', 'n_intervals'),
     Input('data-option', 'value'),
     Input('palette-option', 'value')]
)
def update_slow_graphs(n, data_option, palette_option):
    
    # Update the cached data
    update_cached_data()  # Update cache with new files or lines
    
    # Extract the palette based on the user's choice
    palette = palettes[palette_option]
    
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
    
    # Fetch the indices of images in each condition based on the palette
    palette_idx = {condition: [i for i, img in enumerate(images) if condition in img]
                  for condition in palette.keys()}
    
    # Pre-extract x-coordinates as a NumPy array (n_images × time_range)
    x_coords = np.array([avg_img_trajectories[img]['x_coord'] for img in images])  # shape: (n_images, time_range)

    # Pre-extract ratings for all correlation targets into dictionaries of arrays
    ratings_dict = {
        c: np.array([ratings.loc[ratings['filename'] == img, c].values[0] for img in images])
        for c in correlations.keys()
    }

    # Initialize correlations with nan arrays (for vectorized assignment)
    time_range = x_coords.shape[1]
    # correlations = {c: {p: np.full(time_range, np.nan)}
    correlations = {
        c: {p: np.full(time_range, np.nan)
            for p in palette.keys()}
        for c in correlations.keys()
    }

    # Go over each time point in the time range
    for t in range(time_range):
        x_at_t = x_coords[:, t]  # Extract all x-coordinates at time t

        if not np.any(np.isnan(x_at_t)):  # Check for nan values only once
            for c, target in ratings_dict.items():
                for p in palette.keys(): # loop over the conditions from the palette
                    # find the indices corresponding to the conditions
                    indices = palette_idx[p]
                    correlations[c][p][t] = np.corrcoef(x_at_t[indices], target[indices])[0, 1]  # Compute correlation
    
    # Make a plot from the correlations for each valence type and correlation target
    figs = []
    for valence in valences:
        figs.append(plot_time_correlations(correlations[f'{valence}-valence-gpt'], f'{valence} valence', palette))
    for valence in valences:
        figs.append(plot_time_correlations(correlations[f'{valence}-valence-low'], f'{valence} valence', palette))
    for valence in valences:
        figs.append(plot_time_correlations(correlations[f'{valence}-valence-mid'], f'{valence} valence', palette))
    for valence in valences:
        figs.append(plot_time_correlations(correlations[f'{valence}-valence-high'], f'{valence} valence', palette))
    
    return figs


# Launchin the app
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