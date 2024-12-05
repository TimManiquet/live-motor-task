'''
Load the data from a live mouse tracking task.

author Tim Maniquet
created 2024-11-18
'''

import os, glob, json
from src.utils import *
from plotting.plot_utils import plot_xy_with_boxes
import matplotlib.pyplot as plt

# Declare where the data will be uploaded live
data_dir = './data'

# Find all data files in there
files = glob.glob(os.path.join(data_dir, '*.csv'))

# Find the latest data file there
file = get_latest_file(files)

# Read the data as a dataframe
# df = pd.read_csv(file)
with open(file, mode='r') as file:
    data = json.load(file)

# Extract the last trial
last_trial = data[-1]

# Declare a palette
palette = {
    'negative': 'red',
    'positive': 'blue'
}

# Plot a single xy trajectory
x_coords = last_trial['x_coord']
y_coords = last_trial['y_coord']
condition = last_trial['condition']
col = palette[condition]

# Initiate the handes and labels for the legend

handles = []
labels = []

# Plot the trajectory line for each row
line, = plt.scatter(
    x_coords,
    y_coords,
    label=condition,
    c = col,
    alpha = 0.4,
    # axes = ax,
    # marker = 'o'
)
if condition not in labels:
    handles.append(line)
    labels.append(condition)
plt.legend(handles, labels, bbox_to_anchor=(1.1, 1.0), frameon = False)

plt.show()