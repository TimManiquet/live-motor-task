'''
Plotting utilities for a live mouse tracker demo

author Tim Maniquet
created 2024-11-18
'''

# General imports
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import json

# Load the psychopy elements appearence arguments
boxes_kwargs_file = './src/external/boxes_kwargs.json'
with open(boxes_kwargs_file, 'r') as file:
    boxes_kwargs = json.load(file)

def plot_trajectory_last_trial(last_trial, win_size, palette, show_boxes = True, **kwargs):
    """
    Plot the mouse trajectory of the last trial

    Args:
        last_trial (dict): data from the last trial
    """
    # Declare a scaling factor to adjust the plot size
    scaling_factor = kwargs.get('scaling_factor', 0.9)
    
    # Extract the condition
    # condition = last_trial['condition']
    condition = next((key for key in palette.keys() if key in last_trial['stimulus']), None)
    # Extract the condition colour
    colour = palette[condition]
    
    # Define the plot based on x and y coordinates
    fig = go.Figure(data=go.Scatter(
        x=last_trial['x_coord'],
        y=last_trial['y_coord'],
        mode='lines+markers',
        line = dict(width=2, color=colour)
    ))

    # Adjust the plot elements
    fig.update_layout(
        title = f"Trajectory of the last trial",
        # xaxis_title = "X Coordinates",
        # yaxis_title = "Y Coordinates",
        xaxis = dict(
            range=[- win_size[0] / 2, win_size[0] / 2],
            showgrid=False,  # Hide the grid
            zeroline=False,  # Hide the zero line
            visible=False    # Hide the axis entirely
        ),
        yaxis = dict(
            range=[- win_size[1] / 2, win_size[1] / 2],
            showgrid=False,  # Hide the grid
            zeroline=False,  # Hide the zero line
            visible=False    # Hide the axis entirely
        ),
        width = win_size[0] * scaling_factor,
        height = win_size[1] * scaling_factor,
        plot_bgcolor='white',  # Set the background color of the plot area
        paper_bgcolor='white',  # Set the background color outside the plot area
    )
    
    # If required, plot the mouse tracking visual elements
    if show_boxes:
        # Get the coordinates to draw the boxes on screen
        rectangles = calculate_rectangles(win_size, **boxes_kwargs)
        # Draw the rectangles
        for rect in rectangles:
            fig.add_shape(
                type="rect",
                x0=rect[0],
                y0=rect[1],
                x1=rect[2],
                y1=rect[3],
                line=dict(color="black", width=2), 
                fillcolor="white",
                opacity=0.5
            )
    
    return fig

def plot_cond_trajectories(condition_trajectories, average_trajectories, win_size, palette, show_boxes = True, **kwargs):
    """
    Plot mouse trajectories grouped by conditions.

    Args:
        condition_trajectories (dict): A dictionary where each key represents a condition, 
            and its value is another dictionary with:
                - 'x_coord': A list of lists, where each inner list is the x-coordinates of a trajectory.
                - 'y_coord': A list of lists, where each inner list is the y-coordinates of a trajectory.
    """
    # Declare a scaling factor to adjust the size of the plot
    scaling_factor = kwargs.get('scaling_factor', 0.9)
    
    # Initiate a plotly figure
    fig = go.Figure()
    
    # If required, show the mouse tracking box layout
    if show_boxes:
        # Get the coordinates to draw the boxes on screen
        rectangles = calculate_rectangles(win_size, **boxes_kwargs)
        
        # Draw the rectangles
        for rect in rectangles:
            fig.add_shape(
                type="rect",
                x0=rect[0],
                y0=rect[1],
                x1=rect[2],
                y1=rect[3],
                line=dict(color="black", width=2.0),
                fillcolor="white",
                opacity=0.3
            )

    # Loop through each condition and plot all trajectories
    for condition, trajectories in condition_trajectories.items():
        x_coords = trajectories['x_coord']
        y_coords = trajectories['y_coord']

        for x, y in zip(x_coords, y_coords):
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'{condition} trajectory',
                line=dict(
                    color=palette[condition],
                    width=1.0
                    ),
                opacity = 0.5,
                showlegend=False  # Set to False to avoid cluttering the legend
            ))
    
    # Loop through each condition and plot the average trajectory
    for condition, trajectory in average_trajectories.items():
        x_coords = trajectory['x_coord']
        y_coords = trajectory['y_coord']

        fig.add_trace(go.Scatter(
            x=trajectory['x_coord'],
            y=trajectory['y_coord'],
            mode='lines',
            name=f'{condition} trajectory',
            line=dict(color=palette[condition],width=5.0),
            opacity = 1.0,
            showlegend=False  # Set to False to avoid cluttering the legend
        ))
    
    # Add layout details
    fig.update_layout(
        title="Mouse Trajectories by Condition",
        # xaxis_title="X Coordinates",
        # yaxis_title="Y Coordinates",
        showlegend=True,
        legend_title="Conditions",
        xaxis = dict(
            range=[- win_size[0] / 2, win_size[0] / 2],
            showgrid=False,  # Hide the grid
            zeroline=False,  # Hide the zero line
            visible=False    # Hide the axis entirely
        ),
        yaxis = dict(
            range=[- win_size[1] / 2, win_size[1] / 2],
            showgrid=False,  # Hide the grid
            zeroline=False,  # Hide the zero line
            visible=False    # Hide the axis entirely
        ),
        width = win_size[0] * scaling_factor,
        height = win_size[1] * scaling_factor,
        plot_bgcolor='white',  # Set the background color of the plot area
        paper_bgcolor='white',  # Set the background color outside the plot area
    )

    # Add condition names to the legend by adding invisible traces (1 per condition)
    for condition in condition_trajectories.keys():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name=condition,
            line = dict(color=palette[condition])
        ))
    
    return fig


def calculate_rectangles(win_size, **kwargs):
    '''
    Calculate the x0, x1, y0, y1 coordinates of the psychopy rectangles
    used in the mouse tracking task, based on input coordinates.
    '''
    
    # Fetch the kwargs
    response_box_width = kwargs.get('RESPONSE_BOX_WIDTH', 180)
    response_box_height = kwargs.get('RESPONSE_BOX_HEIGHT', 70)
    start_box_size = kwargs.get('START_BOX_WIDTH', 30)
    from_left = kwargs.get('RESPONSE_BOX_FROM_LEFT', 270)
    from_right = kwargs.get('RESPONSE_BOX_FROM_RIGHT', 270)
    from_top = kwargs.get('RESPONSE_BOX_FROM_TOP', 270)
    from_bottom = kwargs.get('START_BOX_FROM_BOTTOM', 270)
    
    # Find the left and right of the screen
    x_left = - win_size[0] / 2
    x_right = win_size[0] / 2
    
    # Find the top and bottom of the screen
    y_bottom = - win_size[1] / 2
    y_top = win_size[1] / 2
    
    # Create the starting box positions
    start_box_x0 = - start_box_size/2
    start_box_x1 = start_box_size/2
    start_box_y0 = y_bottom + from_bottom + start_box_size/2
    start_box_y1 = y_bottom + from_bottom - start_box_size/2
    # Create the starting box x0, y0, x1, y1
    start_box = [start_box_x0, start_box_y0, start_box_x1, start_box_y1]
    
    # Create the left response box positions
    left_box_x0 = x_left + from_left - response_box_width / 2
    left_box_x1 = x_left + from_left + response_box_width / 2
    left_box_y0 = y_top - from_top + response_box_height / 2
    left_box_y1 = y_top - from_top - response_box_height / 2
    # Create the left response box x0, y0, x1, y1
    left_box = [left_box_x0, left_box_y0, left_box_x1, left_box_y1]
    
    # Create the right response box positions
    right_box_x0 = x_right - from_right - response_box_width / 2
    right_box_x1 = x_right - from_right + response_box_width / 2
    right_box_y0 = y_top - from_top + response_box_height / 2
    right_box_y1 = y_top - from_top - response_box_height / 2
    # Create the right response box x0, y0, x1, y1
    right_box = [right_box_x0, right_box_y0, right_box_x1, right_box_y1]
    
    return start_box, left_box, right_box


def plot_matrices(matrices, time_points):
    # Number of matrices to plot
    n_matrices = len(time_points)
    
    # Declare a title for each subplot
    titles = [f"{t} ms" for t in time_points]
    
    # Create a subplot layout with 1 row and n_matrices columns
    fig = make_subplots(rows=1, cols=n_matrices, subplot_titles=titles)
    
    for i, time_point in enumerate(time_points):
        # Extract the matrix at time point t
        matrix = matrices[time_point]
        # Add the matrix as a heatmap to the corresponding subplot
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale="Greys",  # Choose a colorscale
                colorbar=dict(title="Value") if i == n_matrices - 1 else None,  # Show colorbar only for the last plot
                showscale=False,
            ),
            row=1,
            col=i + 1
        )
    
    # Update layout for spacing and titles
    fig.update_layout(
        height=400,  # Height of the overall figure
        width=200 * n_matrices,  # Adjust width based on the number of matrices
        title_text="Condition differences across time",
        title_x=0.5,  # Center the title
        plot_bgcolor='white',  # Set the background color of the plot area
        paper_bgcolor='white',  # Set the background color outside the plot area
    )
    
    # Update the aspect ratio of each heatmap to make it square
    fig.update_xaxes(scaleanchor="y", constrain="domain", showticklabels=False)  # Constrain the x-axis to match the y-axis scale
    fig.update_yaxes(scaleanchor="x", constrain="domain", showticklabels=False)  # Constrain the y-axis to match the x-axis scale
    
    # Update subplot titles if provided
    if titles:
        fig.update_annotations(font_size=12)
    
    return fig

def custom_legend(palette):
    # Define the rectangles and their properties
    legend_items = [
        {"color": col, "text": label}
        for label, col in palette.items()
    ]
    
    # Define some plotting parameters
    n_labels = len(legend_items)
    center = 0.5  # Center of the figure
    spacing = 0.5  # Spacing between rectangles
    total_height = (n_labels - 1) * spacing
    start_y = center + total_height / 2
    y_positions = [start_y - i * spacing for i in range(n_labels)]
    
    # Create a blank figure
    fig = go.Figure()

    # Add rectangles and text for the custom legend
    for item, y_pos in zip(legend_items, y_positions):
        # Add the rectangle
        fig.add_shape(
            type="rect",
            x0=0.25, x1=0.65,  # Rectangle width (centered horizontally)
            y0=y_pos - 0.15, y1=y_pos + 0.15,  # Rectangle height
            fillcolor=item["color"],
            line=dict(width=0),  # No border
        )
        
        # Add the label text
        fig.add_annotation(
            x=0.7,  # Position of the text, to the right of the rectangle
            y=y_pos,
            text=item["text"],
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=24),
        )

    # Adjust layout
    fig.update_layout(
        width=400,  # Adjust width as needed
        height=300,  # Adjust height as needed
        margin=dict(l=10, r=10, t=10, b=10),  # Tight margins
        xaxis=dict(visible=False),  # Hide x-axis
        yaxis=dict(visible=False),  # Hide y-axis
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig


# def plot_time_correlation(correlation, label):
#     """
#     This function works well for a single correlation trace
#     Plot a time-resolved correlation

#     Args:
#         correlation (np array): correlation across time (one data point per t)
#         time_range (int): time range over which the correlation elapses
#     """
#     # Find the position of the last non nan element to determines the axes
#     last_value_idx = np.where(~np.isnan(correlation))[0][-1]
    
#     # Define the plot based on x and y coordinates
#     fig = go.Figure()
#     #     data=go.Scatter(
#     #         x=(0, time_range), 
#     #         y=correlation,
#     #         mode='lines',
#     #         line=dict(
#     #         color='dodgerBlue',
#     #         width=5.0
#     #         ),
#     #     )
#     # )
#     fig.add_trace(go.Scatter(
#             x=np.arange(last_value_idx),
#             y=correlation,
#             mode='lines',
#             line=dict(
#                 color='dodgerBlue',
#                 width=5.0
#             ),
#             name=label
#         ))

#     # Adjust the plot elements
#     fig.update_layout(
#         title = f'{label} valence',
#         # xaxis_title = "X Coordinates",
#         yaxis_title = "Correlation",
#         xaxis = dict(
#             range=[0, last_value_idx],
#             # showgrid=False,  # Hide the grid
#             # zeroline=False,  # Hide the zero line
#             # visible=False    # Hide the axis entirely
#         ),
#         yaxis = dict(
#             # range=[-0.5, 0.5],
#             # showgrid=False,  # Hide the grid
#             # zeroline=False,  # Hide the zero line
#             # visible=False    # Hide the axis entirely
#         ),
#         # width = win_size[0] * scaling_factor,
#         # height = win_size[1] * scaling_factor,
#         plot_bgcolor='white',  # Set the background color of the plot area
#         paper_bgcolor='white',  # Set the background color outside the plot area
#     )
    
#     return fig

def plot_time_correlations(correlations, label, palette):
    """
    This function works well for several correlations marked by condition
    Plot a time-resolved correlation

    Args:
        correlation (np array): correlation across time (one data point per t)
        time_range (int): time range over which the correlation elapses
    """
    # Find the position of the last non nan element to determines the axes
    try:
        last_value_idx = max([np.where(~np.isnan(correlations[p]))[0][-1] for p in palette.keys()])
    # Debug purposes: if no value can be found in one of the arrays
    except IndexError:
        last_value_idx = 900
    
    # Define the plot based on x and y coordinates
    fig = go.Figure()
    
    # Go condition by condition and add a trace for each correlation
    for cond in palette.keys():    
        fig.add_trace(go.Scatter(
                x=np.arange(last_value_idx),
                y=correlations[cond],
                mode='lines',
                line=dict(
                    color=palette[cond],
                    width=5.0
                ),
                name=cond
            ))

    # Adjust the plot elements
    fig.update_layout(
        title = f'{label}',
        xaxis_title = "Time",
        yaxis_title = "Correlation",
        xaxis = dict(
            range=[0, last_value_idx],
            # showgrid=False,  # Hide the grid
            # zeroline=False,  # Hide the zero line
            # visible=False    # Hide the axis entirely
        ),
        yaxis = dict(
            # range=[-0.5, 0.5],
            # showgrid=False,  # Hide the grid
            # zeroline=False,  # Hide the zero line
            # visible=False    # Hide the axis entirely
        ),
        # width = win_size[0] * scaling_factor,
        # height = win_size[1] * scaling_factor,
        plot_bgcolor='white',  # Set the background color of the plot area
        paper_bgcolor='white',  # Set the background color outside the plot area
    )
    
    return fig


def plot_dim_reduc(coordinates, time_points, labels, palette):
    
    # Check the data size
    assert len(coordinates) == len(time_points), "Mismatch # of time points and coordinates."
    assert len(coordinates[0]) == len(labels), "Mismatch # coordinates and labels."
    
    # Number of scatters to plot
    n_scatters = len(time_points)
    
    # Declare a title for each subplot
    titles = [f"{t} ms" for t in time_points]
    
    # Create a subplot layout with 1 row and n_matrices columns
    fig = make_subplots(rows=1, cols=n_scatters, subplot_titles=titles)
    
    # Loop over pairs of coordinates
    for i, coord in enumerate(coordinates):
        # If the coordinates are empty, make a dot at [0, 0]
        if isinstance(coord, np.ndarray) and not np.isnan(coord).any():
            # Fetch the corresponding time point
            time_point = time_points[i]
            # Extract the conditions from the labels
            conditions = [next((key for key in palette.keys() if key in label), None) for label in labels]
            # Extract corresponding colours from the palette
            colours = [palette[cond] for cond in conditions]
            # Add scatter point
            fig.add_trace(
                go.Scatter(
                    x=coord[:, 0],
                    y=coord[:, 1],
                    mode='markers',
                    name=f'yes {time_point}',  # Label for each time moment
                    marker=dict(
                        size=8,
                        color=colours,  # Color if palette is provided
                        line=dict(width=1, color='DarkSlateGrey')  # Optional border
                    )
                ),
                row=1,
                col=i + 1
            )
    
    # Update layout details
    fig.update_layout(
        title_text="Distance between stimuli",
        title_x=0.5,  # Center the title
        # xaxis_title="PCA Dimension 1",
        # yaxis_title="PCA Dimension 2",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=200 * len(time_points),
        height=400,
    )
        
    # Update the aspect ratio of each heatmap to make it square
    fig.update_xaxes(scaleanchor="y",constrain="domain",showticklabels=False)  # Constrain the x-axis to match the y-axis scale
    fig.update_yaxes(constrain="domain",showticklabels=False)  # Constrain the y-axis to match the x-axis scale
    # fig.update_xaxes(scaleanchor="y", constrain="domain", showticklabels=False)  # Constrain the x-axis to match the y-axis scale
    # fig.update_yaxes(scaleanchor="x", constrain="domain", showticklabels=False)  # Constrain the y-axis to match the x-axis scale
    # Update the aspect ratio of each subplot to make them square
    # for i in range(1, n_scatters + 1):
    #     fig.update_xaxes(showticklabels=True, row=1, col=i)  # Constrain x-axis to match y-axis
    #     fig.update_yaxes(showticklabels=True, row=1, col=i)  # Constrain y-axis to match x-axis

    # fig.show()
    
    # Update subplot titles if provided
    if titles:
        fig.update_annotations(font_size=12)
    
    return fig



def plot_matrices_and_dim_reduc(matrices, time_points, algorithm, labels, palette):
    
    # Number of matrices to plot
    n_matrices = len(time_points)
    
    # Declare a title for each subplot
    titles = [f"{t} ms" for t in time_points]
    
    # Create a subplot layout with 1 row and n_matrices columns
    fig = make_subplots(rows=2, cols=n_matrices, subplot_titles=titles)
    
    for i, time_point in enumerate(time_points):
        ## First: plot the matrix
        # Extract the matrix at time point t
        matrix = matrices[time_point]
        # Add the matrix as a heatmap to the corresponding subplot
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale="Greys",  # Choose a colorscale
                colorbar=dict(title="Value") if i == n_matrices - 1 else None,  # Show colorbar only for the last plot
                showscale=False,
            ),
            row=1,
            col=i + 1
        )
        ## Second: plot the dimensionality reduction coordinates
        # If there is only one line in the matrix or there are nan values in the matrix
        if (np.isnan(matrix).any()) | matrix.shape[0] < 3:
            # Return a single dot at the origin
            coordinates = [0, 0]
        else:
            # Otherwise perform the dimensionality reduction
            coordinates = algorithm.fit_transform(matrix)
        # Extract the conditions from the labels
        conditions = [next((key for key in palette.keys() if key in label), None) for label in labels]
        # Extract corresponding colours from the palette
        colours = [palette[cond] for cond in conditions]
        # Add scatter point
        fig.add_trace(
            go.Scatter(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                mode='markers',
                name=f'{time_point}',  # Label for each time moment
                marker=dict(
                    size=8,
                    color=colours,  # Color if palette is provided
                    line=dict(width=1, color='DarkSlateGrey')  # Optional border
                )
            ),
            row=2,
            col=i + 1
        )
        
    
    # Update layout for spacing and titles
    fig.update_layout(
        height=550,  # Height of the overall figure
        width=200 * n_matrices,  # Adjust width based on the number of matrices
        title_text="Condition differences across time",
        title_x=0.5,  # Center the title
        plot_bgcolor='white',  # Set the background color of the plot area
        paper_bgcolor='white',  # Set the background color outside the plot area
        showlegend=False,
    )
    
    # Update the aspect ratio of each heatmap to make it square
    # fig.update_xaxes(scaleanchor="y", constrain="domain", showticklabels=False)  # Constrain the x-axis to match the y-axis scale
    # fig.update_yaxes(scaleanchor="x", constrain="domain", showticklabels=False)  # Constrain the y-axis to match the x-axis scale
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    # Update subplot titles if provided
    if titles:
        fig.update_annotations(font_size=12)
    
    return fig