import pandas as pd
import json, glob, os
from src.utils import unpack_mouse_data, get_latest_file, smooth_array, extend_fill_trial
import numpy as np
import matplotlib.pyplot as plt

file_ext = '.json'
data_dir = './data'
all_files = glob.glob(os.path.join(data_dir, f'*{file_ext}'))
data_file = get_latest_file(all_files)

with open(data_file, 'r') as file:
    data = json.load(file)

# time_window = 1000

smooth_average_trajectories = True

kwargs = {
    'smoothing_window': 20,
    'trial_duration': 1200
}

unpacked_data = [extend_fill_trial(trial) for trial in data]

conditions = np.unique([trial['condition'] for trial in unpacked_data])

condition_trajectories = {
    condition: {
        'x_coord': [trial['x_coord'][:time_window] for trial in unpacked_data if trial['condition']==condition],
        'y_coord': [trial['y_coord'][:time_window] for trial in unpacked_data if trial['condition']==condition]
    }
    for condition in conditions
}

avg_condition_trajectories = {
    condition: {
        key: np.nanmean(values, axis=0)
        for key, values in condition_trajectories[condition].items()
    }
    for condition in conditions
}

avg_positive_x = avg_condition_trajectories['positive']['x_coord']
avg_positive_y = avg_condition_trajectories['positive']['y_coord']

if smooth_average_trajectories:
    avg_positive_x = smooth_array(avg_positive_x, smoothing_window)
    avg_positive_y = smooth_array(avg_positive_y, smoothing_window)


# plt.plot(avg_positive_x, avg_positive_y, marker='o', linestyle=None)
plt.plot(avg_positive_x, avg_positive_y)
plt.show()

import numpy as np
# Make an image-wise dictionary of trajectories
images = np.unique([trial['stimulus'] for trial in data])
data = unpacked_data
smoothing_window = 20

img_trajectories = {
    img: {
        'x_coord': [trial['x_coord'] for trial in data if trial['stimulus']==img],
        'y_coord': [trial['y_coord'] for trial in data if trial['stimulus']==img]
    }
    for img in images
}

avg_img_trajectories = {
    img: {
        key: smooth_array(np.nanmean(values, axis=0), smoothing_window)
        for key, values in img_trajectories[img].items()
    }
    for img in images
}

