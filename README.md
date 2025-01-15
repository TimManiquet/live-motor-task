# Live motor task

TODO: add a nice illustration of how this would set up on two monitors
### What this is

Two main parts: the Psychopy task and the Dash app.

The Psychopy task is a motor movement experiment. In it, the user is prompted to classify images as positive or negative. These images appear at the center of the screen when trials start, and are rated by clicking on buttons positioned at the top left and top right of the screen.

The Dash app contains plotting tools to visualise the data created during the Psychopy task. It produces a series of graphs, and refreshes frequently to update the data showed on the graphs.

### How to install

Clone this repository

Navigate to the folder

Install dependencies using `conda env create -f environment.yml`.

### How to launch

To run the full project, run `main.py`. This will start both the Psychopy task and the Dash app. You will be prompted to choose a display on which to run the motor task, and the url address of the Dash app will be copied to your clipboard. Open that url on your local browser, and start the motor experiment to generate data to show on the app.

### Repository tree

(TODO: Fix the tree below)

```
live-motor-task
├── app                  # All app-related files (Dash app)
│   ├── app.py           # Main app file
│   ├── assets           # CSS, JS, or other static assets
│   │   └── styles.css
│   └── utils.py         # Helper functions specific to the app
├── experiments          # PsychoPy experiments
│   ├── mouse_tracker_task.py  # Main experiment file
│   ├── mock_images.py   # Supporting experiment script
│   └── dont_delete.py   # Supporting experiment script
├── data                 # All experiment data
│   ├── 2024-12-23-1627_task-mockmousetracker.json
│   └── (other .json files)
├── plotting             # Data visualization utilities
│   ├── plot_utils.py    # Core plotting functions
├── src                  # Core utilities shared by app and experiments
│   ├── __init__.py
│   ├── psychopy_utils.py
│   ├── utils.py
│   └── external         # External data or configs
│       ├── boxes_kwargs.json
│       └── lists.json
├── stimuli              # Stimuli used in experiments
│   ├── beach_ambiguous_interaction.jpg
│   ├── (other image files)
├── config.py            # Centralized configuration (paths, settings, etc.)
├── main.py              # Central entry point to launch both app and experiment
└── README.md            # Project overview and instructions
```


### Functionalities

self copy-paste after launching

the task: re-launching, not skipping trials, escaping