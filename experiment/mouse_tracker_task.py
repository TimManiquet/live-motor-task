"""
Experimental task with recording of mouse coordinates across time.
Developped to save data "live" after trial for the sake of a demo.

author Tim Maniquet
created 4 November 2024
"""

# GLOBAL IMPORTS
from psychopy import visual, event, core
import os, json, pyglet, random
import numpy as np

# ADD SOURCE TO THE PATH
import sys
sys.path.append(".")

# LOCAL IMPORTS
from src.utils import *
from src.psychopy_utils import press_effect, select_monitor, restart_animation
from src import palette

## GLOBAL PARAMETERS

# Debugging mode
DEBUGGING_MODE = False
DEBUGGING_LENGTH = 5 # How many trials to play in debugging mode

# Experiment window parameters
SELECT_MONITOR = True # prompt the user to choose a screen or not
FULLSCREEN_MODE = False
WIN_WIDTH = 1200
WIN_HEIGHT = 800
BACKGROUND_COLOUR = 1.0 # 1.0 for white

# Durations
STIMULUS_DURATION = 0.5  # duration in seconds
ITI = 0.5
TRIAL_DURATION = 1.2
RESTART_DURATION = 2.0 # waiting time when restarting

# Stimulus
STIM_WIDTH = 700 # stimulus width
STIM_HEIGHT = 500 # stimulus height

# Load the box appearence keywords
boxes_kwargs_file = './src/external/boxes_kwargs.json'
with open(boxes_kwargs_file, 'r') as file:
    boxes_kwargs = json.load(file)

# Extract the values from it
# Box colours
BOX_FILLCOLOUR = boxes_kwargs['BOX_FILLCOLOUR']
BOX_EDGECOLOUR = boxes_kwargs['BOX_EDGECOLOUR']

# Starting box
START_BOX_FROM_BOTTOM = boxes_kwargs['START_BOX_FROM_BOTTOM']
START_BOX_WIDTH = boxes_kwargs['START_BOX_WIDTH']
START_BOX_HEIGHT = boxes_kwargs['START_BOX_HEIGHT']

# Response boxes
RESPONSE_BOX_FROM_LEFT = boxes_kwargs['RESPONSE_BOX_FROM_LEFT']
RESPONSE_BOX_FROM_RIGHT = boxes_kwargs['RESPONSE_BOX_FROM_RIGHT']
RESPONSE_BOX_FROM_TOP = boxes_kwargs['RESPONSE_BOX_FROM_TOP']
RESPONSE_BOX_HEIGHT = boxes_kwargs['RESPONSE_BOX_HEIGHT']
RESPONSE_BOX_WIDTH = boxes_kwargs['RESPONSE_BOX_WIDTH']

# Define the possible responses
RESPONSE_1 = "positive"
RESPONSE_2 = "negative"

# Declare the output dire
out_dir = 'data'

# Give a name to the task & take a timestamp
task_name = 'mockmousetracker'

# Extract condition names from the palette
conditions = list(palette.keys())


## WINDOW CREATION

# If required, ask the user to select a monitor
if SELECT_MONITOR:
    # Fetch the available monitors
    display = pyglet.canvas.get_display()
    available_monitors = display.get_screens()
    # Show a dialog box to choose the monitor to display the task on
    selected_monitor_idx = select_monitor(available_monitors)

# Create the main PsychoPy window on the selected monitor
win = visual.Window(
    screen = selected_monitor_idx if SELECT_MONITOR else 0,  # Index of the monitor
    fullscr = FULLSCREEN_MODE,  # Fullscreen
    units='pix',  # Use pixels as the unit,
    size = (WIN_WIDTH, WIN_HEIGHT),
    color = BACKGROUND_COLOUR,
    allowGUI = None
)

# Extract the window size
WIN_WIDTH, WIN_HEIGHT = win.size[0], win.size[1]

## To cope with retina displays: get the pyglet pixel ratio

# Extract the underlying pyglet window
pyglet_window = win.winHandle
# Find the corresponding pixel ratio (higher in retina displays)
pixel_ratio = pyglet_window.get_pixel_ratio()
# Scale the window size accordingly
WIN_WIDTH /= pixel_ratio
WIN_HEIGHT /= pixel_ratio

# Calculate the boxes positions
START_BOX_POS = (0, -WIN_HEIGHT / 2 + START_BOX_FROM_BOTTOM)
CHOICE_BOX_POS_LEFT = (-WIN_WIDTH / 2 + RESPONSE_BOX_FROM_LEFT, WIN_HEIGHT / 2 - RESPONSE_BOX_FROM_TOP)
CHOICE_BOX_POS_RIGHT = (WIN_WIDTH / 2 - RESPONSE_BOX_FROM_RIGHT, WIN_HEIGHT / 2 - RESPONSE_BOX_FROM_TOP)

## STIMULI & VISUAL ELEMENTS

# Import stimuli and list them based on the debugging mode
stimuli_dir = './stimuli'
if not DEBUGGING_MODE:
    image_files = [os.path.join(stimuli_dir, img) for img in os.listdir(stimuli_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
elif DEBUGGING_MODE:
    image_files = [os.path.join(stimuli_dir, img) for img in os.listdir(stimuli_dir) if img.endswith(('.png', '.jpg', '.jpeg'))][:DEBUGGING_LENGTH+1]

# Define an ImageStim for displaying images
STIMULUS_IMAGE = visual.ImageStim(win, image=None, pos=(0, 0), size=(STIM_WIDTH, STIM_HEIGHT))

# Define visual screen components
START_BOX = visual.Rect(win, width=START_BOX_WIDTH, height=START_BOX_HEIGHT, pos=START_BOX_POS, fillColor=BOX_FILLCOLOUR, lineColor=BOX_EDGECOLOUR)
START_TEXT = visual.TextStim(win, text="+", pos=START_BOX_POS, color="black")
CHOICE_BOX_LEFT = visual.Rect(win, width=RESPONSE_BOX_WIDTH, height=RESPONSE_BOX_HEIGHT, pos=CHOICE_BOX_POS_LEFT, fillColor=BOX_FILLCOLOUR, lineColor=BOX_EDGECOLOUR)
CHOICE_BOX_RIGHT = visual.Rect(win, width=RESPONSE_BOX_WIDTH, height=RESPONSE_BOX_HEIGHT, pos=CHOICE_BOX_POS_RIGHT, fillColor=BOX_FILLCOLOUR, lineColor=BOX_EDGECOLOUR)
CHOICE_TEXT_LEFT = visual.TextStim(win, text=RESPONSE_1, pos=CHOICE_BOX_POS_LEFT, color="black")
CHOICE_TEXT_RIGHT = visual.TextStim(win, text=RESPONSE_2, pos=CHOICE_BOX_POS_RIGHT, color="black")

# Define restarting component
RESTART_TEXT = visual.TextStim(win, text="Restarting experiment ...", pos=(0,0), color="black")

# Initiate a trial data list
trial_data = []

# Initiate a mouse object
mouse = event.Mouse(visible=True, win=win)
# The next line is a trick to reset the pressing state of the button
event.mouseButtons = [0, 0, 0]  # Reset button states

# Initiate the reaction time clock
rt_clock = core.Clock()  # Reaction time clock

# Start a timer to record mouse data at a certain rate
mouse_timer = core.Clock()
SAMPLING_RATE = 0.01 # sampling rate of 1ms


while True:
    # Shuffle the order of the stimuli
    random.shuffle(image_files)
    
    # Main experiment loop
    for i, image_path in enumerate(image_files):
        
        # Set the image to be displayed
        STIMULUS_IMAGE.image = image_path
        # Extract the stimulus label & condition
        stimulus_label = os.path.basename(image_path)
        stimulus_condition = [cond for cond in conditions if cond in stimulus_label][0]
        # Start waiting for an answer
        response_given = False
        
        while not response_given:
            # Initiate the mouse recording
            x_coordinates = []
            y_coordinates = []
            mouse_times = []

            # Show the trial start box            
            start_trial = False
            while not start_trial:
                # Draw start box
                START_BOX.draw()
                START_TEXT.draw()
                win.flip()
                
                # Check for key presses
                keys_event = event.getKeys()
                if keys_event:
                    # Check for a restart signal
                    if 'r' in keys_event:
                        restart_animation(win, RESTART_DURATION, RESTART_TEXT)
                        break # exit the loop to go back to the main experiment loop
                    # Check for Escape key press
                    elif 'escape' in keys_event:
                        # Close everything
                        win.close()
                        core.quit()
                
                # If the start box is pressed
                if mouse.isPressedIn(START_BOX):
                    # Declare the start of the trial
                    start_trial = True
                    # Show the effect of pressing the start box
                    press_effect(win, START_BOX, START_TEXT)
                    win.flip()
                    # Start the RT clock
                    rt_clock.reset()
                
            ## Start the trial 
            
            # Start the trial duration clock
            trial_clock = core.Clock()
            # Reset the mouse timer
            mouse_timer.reset()
            
            # Start recording mouse data
            while (trial_clock.getTime() < TRIAL_DURATION) & (not response_given):
                
                if 'r' in keys_event:  # Ensure we exit the outer loop too if restarting
                    break
                
                # Log mouse position at the given frame rate
                if mouse_timer.getTime() >= SAMPLING_RATE:
                    mouse_times.append(np.round(rt_clock.getTime() * 1000))
                    x_coordinates.append(mouse.getPos()[0])
                    y_coordinates.append(mouse.getPos()[1])
                    mouse_timer.reset()

                # Show the image for a set duration
                if trial_clock.getTime() <= STIMULUS_DURATION:
                    # Draw the stimulus
                    STIMULUS_IMAGE.draw()
                    win.flip()
                
                # Otherwise if we passed the stimulus duration, show the boxes
                elif trial_clock.getTime() > STIMULUS_DURATION:
                    # Draw the response boxes only
                    CHOICE_BOX_LEFT.draw()
                    CHOICE_BOX_RIGHT.draw()
                    CHOICE_TEXT_LEFT.draw()
                    CHOICE_TEXT_RIGHT.draw()
                    win.flip()
                    
                    # Check for an exit press
                    if 'escape' in event.getKeys():
                        # Close everything
                        win.close()
                        core.quit()
                    
                    # Check for responses
                    if mouse.isPressedIn(CHOICE_BOX_LEFT):
                        # Play the clicking effect
                        press_effect(win, CHOICE_BOX_LEFT, CHOICE_TEXT_LEFT)
                        win.flip()
                        # Record the last mouse cursor position
                        mouse_times.append(np.round(rt_clock.getTime() * 1000))
                        x_coordinates.append(mouse.getPos()[0])
                        y_coordinates.append(mouse.getPos()[1])
                        # Record the response
                        response = CHOICE_TEXT_LEFT.text
                        # Declare that a response has been given
                        response_given = True
                    elif mouse.isPressedIn(CHOICE_BOX_RIGHT):
                        # Play the clicking effect
                        press_effect(win, CHOICE_BOX_RIGHT, CHOICE_TEXT_RIGHT)
                        win.flip()
                        # Record the last mouse cursor position
                        mouse_times.append(np.round(rt_clock.getTime() * 1000))
                        x_coordinates.append(mouse.getPos()[0])
                        y_coordinates.append(mouse.getPos()[1])
                        # Record the response
                        response = CHOICE_TEXT_RIGHT.text
                        # Declare that a response has been given
                        response_given = True
                    elif (not response_given) & (trial_clock.getTime() > TRIAL_DURATION):
                        print("time is over and no answer has been given: restarting trial.")
                        break
        
            if 'r' in keys_event:  # Ensure we exit the outer loop too if restarting
                break

        if 'r' in keys_event:  # Ensure we exit the outer loop too if restarting
            break

        # Record accuracy and RT
        reaction_time = rt_clock.getTime()
        accuracy = 1 if response == stimulus_condition else 0
        
        # Save the trial data
        trial_data.append({
            "trial_number": i,
            "stimulus": stimulus_label,
            "condition": stimulus_condition,
            "response": response,
            "rt": reaction_time,
            "accuracy": accuracy,
            "x_coord": x_coordinates,
            "y_coord": y_coordinates,
            "mouse_times": mouse_times
        })

        ## Live updating: save the file after each trial
        # Create a timestamp to save a unique data file
        timestamp = get_current_datetime()
        # Declare output file name
        out_filename = f'{timestamp}_task-{task_name}.json'
        # Save the data to json
        with open(os.path.join(out_dir, out_filename), mode='w') as file:
            json.dump(trial_data, file, indent=4)

        # Wait for the inter trial interval
        core.wait(ITI)

# Close the window at the end
win.close()
core.quit()
