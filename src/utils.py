'''
Utility functions for the mouse tracker mock task and its live analysis.

author Tim Maniquet
created 2024-11-17
'''

import numpy as np
import os, datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_current_datetime():
    """
    Return a string with YYYY-MM-DD-HHMM format.
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d-%H%M')
    return formatted_datetime


def get_latest_file(list_of_files):
    """
    Find the latest file in a list of files.
    """
    return max(list_of_files, key=os.path.getctime)


# def unpack_mouse_data(mouse_coordinates, mouse_times, trial_duration):
#     '''
#     Take in incomplete mouse coordinates and corresponding timings,
#     return extrapolated coordinates and timing in a df format.
#     '''
#     # Extract the x and y coordinates from the mouse data
#     x_coord = mouse_coordinates[:,0]
#     y_coord = mouse_coordinates[:,1]
#     # Extrapolate the empty data points in both
#     extended_x_coord = extend_array(x_coord, mouse_times, trial_duration)
#     extended_y_coord = extend_array(y_coord, mouse_times, trial_duration)
#     trial_df = pd.DataFrame({
#         'x_coord': fill_in_array(extended_x_coord),
#         'y_coord': fill_in_array(extended_y_coord),
#         'time': range(trial_duration),
#     })
#     return trial_df

def extend_fill_trial(trial, trial_duration = 2000):
    '''
    This version works for dictionaries
    Take the data from one trial in a dictionary form, and return it with the
    mouse data extended and filled in.
    The trial duration is set very high to avoid errors in case the trajectory 
    is very long for some reason.
    '''
    x_extended = extend_array(trial['x_coord'], trial['mouse_times'], trial_duration)
    trial['x_coord'] = fill_in_array(x_extended)
    y_extended = extend_array(trial['y_coord'], trial['mouse_times'], trial_duration)
    trial['y_coord'] = fill_in_array(y_extended)
    return trial

# def extend_fill_trial_row(row, trial_duration=2000):
#     '''
#     Process a row of the DataFrame, extend and fill mouse data for x and y coordinates.
#     '''
#     row['x_coord'] = fill_in_array(extend_array(row['x_coord'], row['mouse_times'], trial_duration))
#     row['y_coord'] = fill_in_array(extend_array(row['y_coord'], row['mouse_times'], trial_duration))
#     return row


def extend_array(input_array, indices, length):
    """
    Extend an array
    ---------------
    Takes in an input array with corresponding indices. Returns
    an array of desired length with the input array values 
    placed at the input indices. Rest of the output array is 
    NaN values.

    Args:
        input_array (np array): input values.
        indices (np array): input indices to place the values.
        length (int): desired length of the output array.

    Returns:
        array: the resulting length-long array with input 
            values placed at the input indices.
    """
    # check if there is something at all
    if len(input_array) > 0:
        # make an empty array with the desired length
        array = np.empty(length)
        array[:] = np.nan # fill it with NaNs
        # add the existing values at their correct spots
        array[[int(item) for item in indices]] = input_array
    # otherwise return an array of nan values
    else:
        array = np.empty(length)
        array[:] = np.nan
    
    return array

def fill_in_array(input_array):
    """
    Fill in an array
    ----------------
    Fills in the empty values of an array using linear
    interpolation. Rules:
    - empty values at the start are replaced by the first
      existing value.
    - empty values at the end are left empty.
    - all other empty values are filled in linearly based
      on the upper and lower existing values.

    Args:
        input_array (np array): the array to fill in.

    Returns:
        new array: the resulting array with filled in NaN
            values.
    """
    
    # make a copy of the input array
    new_array = input_array.copy()
        
    # find the existing values and their indices
    existing_indices = np.where(~np.isnan(new_array))[0]

    # find the empty value indices
    nan_indices = np.where(np.isnan(new_array))[0]
    
    # check if the array is empty
    if len(existing_indices) == 0:
        return new_array # just return the empty array
        
    # loop through the empty values and interpolate
    for index in nan_indices:
        # stop filling in if there are no more existing values after    
        if index > np.max(existing_indices):
            continue
        # if there is no existing value before, use the first existing value
        if index < np.min(existing_indices):
            # new_array[index] = existing_values[0]
            continue
        # otherwise perform the interpolation
        else:
            # find the lower existing value
            lower = existing_indices[existing_indices < index][-1]
            upper = existing_indices[existing_indices > index][0]
            new_array[index] = np.interp(index, [lower, upper], [new_array[lower], new_array[upper]])

    return new_array


def smooth_array(data, window_size):
    """
    Smooth a 1D NumPy array containing NaN values using a sliding window.

    Args:
        data (numpy.ndarray): Input 1D array.
        window_size (int): Size of the smoothing window.

    Returns:
        numpy.ndarray: Smoothed array with NaN values preserved.
    """
    # Convert the input array to a Pandas Series for rolling window
    series = pd.Series(data)
    
    # Use rolling window with a mean that skips NaNs
    smoothed = series.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Convert back to NumPy array
    return smoothed.to_numpy()



def unpack_cond_trajectories(unpacked_data, palette, smoothing=True, **kwargs):
    """
    This function works well for a  dictionary
    Processes unpacked mouse tracking data to extract and optionally smooth condition-specific trajectories.

    This function unpacks trial data, groups trajectories by condition, and computes 
    the average trajectory for each condition. Smoothing can be applied to the averaged 
    trajectories if specified.

    Args:
        data (list): A list of trial data dictionaries. Each dictionary should include:
            - 'x_coord' (list): The x-coordinates of the mouse trajectory.
            - 'y_coord' (list): The y-coordinates of the mouse trajectory.
            - 'condition' (str): The condition associated with the trial.
        smoothing (bool): If `True`, smooth the averaged trajectories using a smoothing window.
        **kwargs:
            smoothing_window (int): Size of the smoothing window for the trajectories. 
                Defaults to 20.
            trial_duration (int): Length to which trials should be padded or truncated. 
                Defaults to 1200.

    Returns:
        tuple: A tuple containing:
            - `condition_trajectories` (dict): Trajectories grouped by condition. Each key 
              represents a condition, and its value is a dictionary with keys 'x_coord' and 
              'y_coord', containing lists of trajectories.
            - `avg_condition_trajectories` (dict): Averaged and smoothed (if enabled) 
              trajectories for each condition. Each key represents a condition, and its 
              value is a dictionary with keys 'x_coord' and 'y_coord', containing the 
              averaged trajectories.
    """
    
    # Extract the keyword arguments
    smoothing_window = kwargs.get('smoothing_window', 20)
    # If no smoothing is requested, set the smoothing window to 1
    if not smoothing:
        smoothing_window = 1
    
    # Find out all the unique conditions
    # conditions = np.unique([trial['condition'] for trial in unpacked_data])
    conditions = list(palette.keys())
    # Extract the trajectories for each condition
    condition_trajectories = {
        condition: {
            'x_coord': [trial['x_coord'] for trial in unpacked_data if condition in trial['stimulus']],
            'y_coord': [trial['y_coord'] for trial in unpacked_data if condition in trial['stimulus']]
            # 'x_coord': [trial['x_coord'] for trial in unpacked_data if trial['condition']==condition],
            # 'y_coord': [trial['y_coord'] for trial in unpacked_data if trial['condition']==condition]
        }
        for condition in conditions
    }
    # Make an average xy trajectory per condition
    avg_condition_trajectories = {
        condition: {
            key: smooth_array(np.nanmean(values, axis=0), smoothing_window)
            for key, values in condition_trajectories[condition].items()
        }
        for condition in conditions
    }

    return condition_trajectories, avg_condition_trajectories

# def unpack_cond_trajectories(df, smoothing=True, **kwargs):
#     """
#     This function works well for a dataframe
#     Processes mouse tracking data in a DataFrame to extract and optionally smooth condition-specific trajectories.

#     Args:
#         df (pd.DataFrame): A DataFrame containing trial data with columns:
#             - 'x_coord' (list): The x-coordinates of the mouse trajectory.
#             - 'y_coord' (list): The y-coordinates of the mouse trajectory.
#             - 'condition' (str): The condition associated with the trial.
#         smoothing (bool): If `True`, smooth the averaged trajectories using a smoothing window.
#         **kwargs:
#             smoothing_window (int): Size of the smoothing window for the trajectories. Defaults to 20.
#             trial_duration (int): Length to which trials should be padded or truncated. Defaults to 1200.

#     Returns:
#         tuple: (condition_trajectories, avg_condition_trajectories)
#             - `condition_trajectories` (dict): Trajectories grouped by condition.
#             - `avg_condition_trajectories` (dict): Averaged and smoothed (if enabled) trajectories for each condition.
#     """
    
#     # Extract the keyword arguments
#     smoothing_window = kwargs.get('smoothing_window', 20)
#     if not smoothing:
#         smoothing_window = 1

#     # Group the data by the 'condition' column
#     condition_groups = df.groupby('condition')

#     # Extract trajectories for each condition
#     condition_trajectories = {
#         condition: {
#             'x_coord': list(group['x_coord']),
#             'y_coord': list(group['y_coord'])
#         }
#         for condition, group in condition_groups
#     }

#     # Compute average trajectory for each condition
#     avg_condition_trajectories = {
#         condition: {
#             'x_coord': smooth_array(np.nanmean(np.vstack(values['x_coord']), axis=0), smoothing_window),
#             'y_coord': smooth_array(np.nanmean(np.vstack(values['y_coord']), axis=0), smoothing_window)
#         }
#         for condition, values in condition_trajectories.items()
#     }

#     return condition_trajectories, avg_condition_trajectories


def unpack_img_trajectories(unpacked_data, smoothing=True, **kwargs):
    '''
    This function works well for dictionaries.
    Takes in unpacked raw data and returns image-wise trajectories and
    image-wise average trajectories.
    '''
    # Extract the keyword arguments
    smoothing_window = kwargs.get('smoothing_window', 20)
    # If no smoothing is requested, set the smoothing window to 1
    if not smoothing:
        smoothing_window = 1
    # Fetch all the unique images
    images = np.unique([trial['stimulus'] for trial in unpacked_data])
    # Aggregate all the trajectories per image
    img_trajectories = {
        img: {
            'x_coord': [trial['x_coord'] for trial in unpacked_data if trial['stimulus']==img],
            'y_coord': [trial['y_coord'] for trial in unpacked_data if trial['stimulus']==img]
        }
        for img in images
    }
    # Average over images
    avg_img_trajectories = {
        img: {
            key: smooth_array(np.nanmean(values, axis=0), smoothing_window)
            for key, values in img_trajectories[img].items()
        }
        for img in images
    }
    
    return img_trajectories, avg_img_trajectories

# def unpack_img_trajectories(df, smoothing=True, **kwargs):
#     '''
#     This function works well for a dataframe
#     Takes in mouse tracking data in a DataFrame and returns image-wise trajectories 
#     and image-wise average trajectories.

#     Args:
#         df (pd.DataFrame): A DataFrame with columns:
#             - 'x_coord' (list): The x-coordinates of the mouse trajectory.
#             - 'y_coord' (list): The y-coordinates of the mouse trajectory.
#             - 'stimulus' (str): The image associated with the trial.
#         smoothing (bool): If `True`, smooth the averaged trajectories using a smoothing window.
#         **kwargs:
#             smoothing_window (int): Size of the smoothing window for the trajectories. Defaults to 20.

#     Returns:
#         tuple: (img_trajectories, avg_img_trajectories)
#             - `img_trajectories`: Trajectories grouped by image.
#             - `avg_img_trajectories`: Averaged and smoothed (if enabled) trajectories for each image.
#     '''
#     # Extract the keyword arguments
#     smoothing_window = kwargs.get('smoothing_window', 20)
#     if not smoothing:
#         smoothing_window = 1

#     # Group the data by the 'stimulus' column
#     img_groups = df.groupby('stimulus')

#     # Aggregate all the trajectories per image
#     img_trajectories = {
#         img: {
#             'x_coord': list(group['x_coord']),
#             'y_coord': list(group['y_coord'])
#         }
#         for img, group in img_groups
#     }

#     # Compute average trajectory for each image
#     avg_img_trajectories = {
#         img: {
#             'x_coord': smooth_array(np.nanmean(np.vstack(values['x_coord']), axis=0), smoothing_window),
#             'y_coord': smooth_array(np.nanmean(np.vstack(values['y_coord']), axis=0), smoothing_window)
#         }
#         for img, values in img_trajectories.items()
#     }

#     return img_trajectories, avg_img_trajectories


def calculate_pairwise_diff_matrices(avg_img_trajectories):
    '''
    Takes in average image trajectories and returns the x position
    pairwise difference matrices.
    '''
    image_names = list(avg_img_trajectories.keys())
    x_coords = [avg_img_trajectories[name]['x_coord'] for name in image_names]

    # Stack into a 2D array (time_points x images)
    x_coords = np.array(x_coords).T  # Shape: (time_points, images)

    # Initialize a list to store the pairwise difference matrices
    difference_matrices = []

    # Loop through each time point
    for t in range(x_coords.shape[0]):
        # Extract x values at this time point for all images
        x_at_t = x_coords[t, :]  # Shape: (images,)
        # Compute pairwise differences
        diff_matrix = np.abs(x_at_t[:, np.newaxis] - x_at_t[np.newaxis, :])  # Shape: (images, images)
        # Append the matrix to the list
        difference_matrices.append(diff_matrix)
    
    return difference_matrices


def reorder_matrices_by_conditions(difference_matrices, image_names, conditions):
    """
    Reorder the matrices based on conditions in the image names.
    
    Args:
        difference_matrices (list of np.ndarray): Pairwise difference matrices (1 per time point).
        image_names (list of str): Names of the images.
        conditions (list of str): Order of conditions to sort by.
    
    Returns:
        list of np.ndarray: Reordered matrices.
        list of str: Reordered image names.
    """
    # Map image names to their conditions
    image_conditions = [next((cond for cond in conditions if cond in name), None) for name in image_names]
    
    # Ensure all images have a condition
    if None in image_conditions:
        raise ValueError(f"Some images do not match any condition: {image_names}")
    
    # Sort indices based on the order of conditions
    sorted_indices = sorted(range(len(image_names)), key=lambda i: conditions.index(image_conditions[i]))
    
    # Reorder image names
    reordered_names = [image_names[i] for i in sorted_indices]
    
    # Reorder matrices
    reordered_matrices = []
    for matrix in difference_matrices:
        reordered_matrix = matrix[np.ix_(sorted_indices, sorted_indices)]
        reordered_matrices.append(reordered_matrix)
    
    return reordered_matrices, reordered_names

def dim_reduc_matrices(matrices, time_points, algorithm, standardise = True):
    
    # Initiate an empty list to hold onto the results
    dim_reduc_coordinates = []
    
    # Loop over time points and extract the results
    for time_point in time_points:
        # Extract the corresponding matrix
        matrix = matrices[time_point]
        # Check if there are empty data points in the matrix
        if np.isnan(matrix).any():
            dim_reduc_coordinates.append(np.nan)
            continue
        # Fit the dimensionality reduction to the matrix
        coordinates = algorithm.fit_transform(matrix)
        dim_reduc_coordinates.append(coordinates)
        
        # # If required, standardise the results
        # if standardise:
        #     std_coordinates = (coordinates - coordinates.mean())/(coordinates.std())
        #     # Store the resulting coordinates after standardising
        #     dim_reduc_coordinates.append(std_coordinates)
        # else:
        #     # Store the resulting coordinates without standardising
        #     dim_reduc_coordinates.append(coordinates)
    # If necessary, standardise the results
    if standardise:
        # Turn the list to an array
        array = np.array(dim_reduc_coordinates)
        # Standardise the array
        array = (array - array.mean()/array.std())
        # Turn it back into a list
        dim_reduc_coordinates = list(array)
    
    return dim_reduc_coordinates