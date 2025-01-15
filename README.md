# Live motor task

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