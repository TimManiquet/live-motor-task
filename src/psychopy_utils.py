'''
Utility functions for the mouse tracker live task, in particular
elements for psychopy.

author Tim Maniquet
created 2024-11-28
'''

from psychopy import visual, core, event
# import tkinter as tk
# from tkinter import simpledialog


def press_effect(win, box, text, pressed_box_color="lightGray", pressed_text_color="Gray", duration=0.03):
    """
    Simulate a pressing effect on a button and its associated text.

    Parameters:
        box (visual.Rect): The button rectangle to be modified.
        text (visual.TextStim): The text object associated with the button.
        pressed_box_color (str or tuple, optional): The fill color to use during the press effect. Defaults to "gray".
        original_text_color (str or tuple, optional): The original color of the text. Defaults to "black".
        pressed_text_color (str or tuple, optional): The text color to use during the press effect. Defaults to "white".
        duration (float, optional): The duration of the press effect in seconds. Defaults to 0.2.
    """
    # Extract the original values
    original_box_color = box.fillColor
    original_text_color = text.color
    
    # Change to pressed colors
    box.fillColor = pressed_box_color
    text.color = pressed_text_color
    
    # Draw the updated box and text
    box.draw()
    text.draw()
    win.flip()
    core.wait(duration)
    
    # Restore the values
    box.fillColor = original_box_color
    text.color = original_text_color



def select_monitor(monitors):
    """
    Creates a PsychoPy window with buttons representing monitors, and closes when a button or ESC is pressed.
    
    Parameters:
        monitors (list): A list of monitor objects with attributes x, y, width, and height.
    
    Returns:
        int: Index of the selected monitor, or prints a message if ESC is pressed.
    """
    # Constants for button and window layout
    padding = 50  # Space between buttons
    button_spacing = 30  # Space between buttons
    scale_factor = 0.15  # Scaling factor for monitor rectangles
    instruction_text = "Please select a monitor to display the experiment on:"

    # Calculate overall window size
    total_width = sum(mon.width * scale_factor for mon in monitors) + (len(monitors)) * button_spacing + padding
    max_height = max(mon.height * scale_factor for mon in monitors) + 150  # Include room for text
    win_size = (int(total_width), int(max_height))
    
    # Create a PsychoPy window without decorations
    win = visual.Window(
        size=win_size,
        fullscr=False,
        screen=0,
        units="pix",
        color="white",
        allowGUI=False,  # Remove top bar
    )
    # Declare the mouse element
    mouse = event.Mouse(visible=True, win=win)

    # Instruction text
    instruction = visual.TextStim(
        win,
        text=instruction_text,
        pos=(0, win_size[1] // 2 - 50),  # Place near the top
        color="black",
        height=24,
        wrapWidth=total_width - 20,
    )

    # Draw monitor buttons
    buttons = []
    # x_pos = -total_width // 2 + monitors[0].width * scale_factor / 2 + padding // 2 # Start position for buttons
    x_pos = -total_width // 2 + padding // 2 # Start position for buttons
    for idx, mon in enumerate(monitors):
        # Initiate the x position at the horizontal center of the screen
        x_pos +=  mon.width * scale_factor / 2
        button = visual.Rect(
            win,
            width=mon.width * scale_factor,
            height=mon.height * scale_factor,
            fillColor="white",
            lineColor="black",
            pos=(x_pos, 0),
        )
        label = visual.TextStim(
            win,
            text=f"Monitor {idx + 1}\n{mon.width}x{mon.height}",
            pos=(x_pos, -mon.height * scale_factor / 2 - 20),
            height=20,
            wrapWidth=mon.width * scale_factor - 10,
            color="black",
        )
        buttons.append((button, label))
        # Add the button spacing before the next rectangle is draw
        x_pos += button_spacing + mon.width * scale_factor / 2

    # Main event loop
    while True:
        # Draw everything
        win.flip()
        instruction.draw()
        for button, label in buttons:
            button.draw()
            label.draw()

        # Wait for a mouse press in one of the two boxes
        for idx, (button, label) in enumerate(buttons):    
            if mouse.isPressedIn(button):
                # Play the clicking effect
                press_effect(win, button, label, pressed_text_color="black")
                # Close the window & return the requested monitor index
                # win.flip()
                win.close()
                # Reset the mouse
                mouse.clickReset()
                return idx
    
        # Check for an escape key press
        keys = event.getKeys()
        if "escape" in keys:
            win.close()
            print("No monitor was selected.")
            return None
    
        
    # # Main event loop
    # selected_monitor_idx = None
    # while selected_monitor_idx is None:
    #     # Draw everything
    #     instruction.draw()
    #     for button, label in buttons:
    #         button.draw()
    #         label.draw()
    #     win.flip()

    #     # Check for a mouse press
    #     for idx, (button, label) in enumerate(buttons):
    #         if mouse.isPressedIn(button):
    #             selected_monitor_idx = idx
    #             break

    #     # Check for escape key press
    #     keys = event.getKeys()
    #     if "escape" in keys:
    #         selected_monitor_idx = None
    #         break

    # Hide the window instead of closing it
    # win.setVisible(False)  # Keep the window for re-use



# def select_monitor(monitors):
#     """
#     Display a dialog box to choose between monitors, showing their relative sizes as rectangles.
    
#     Args:
#         monitors (list): List of monitor objects with attributes `x`, `y`, `width`, and `height`.

#     Returns:
#         int: Index of the selected monitor.
#     """
#     class MonitorSelectionDialog(simpledialog.Dialog):
#         def __init__(self, parent, title, monitors):
#             self.monitors = monitors
#             self.selected_monitor = None
#             super().__init__(parent, title)

#         def body(self, frame):
#             # Add explanatory text
#             label = tk.Label(frame, text="Which monitor would you like to run the experiment on?", font=("Arial", 12))
#             label.pack(pady=10)
            
#             # Determine scaling factor for drawing
#             max_width = max(m.width for m in self.monitors)
#             max_height = max(m.height for m in self.monitors)
#             scale = min(200 / max_width, 200 / max_height)  # Adjust scaling to fit within 200px height

#             # Calculate total width for all monitors side by side
#             padding = 20
#             total_width = sum(m.width * scale for m in self.monitors) + padding * (len(self.monitors) - 1)
#             canvas_width = max(total_width, 600)  # Ensure a minimum width for the canvas
#             canvas_height = 300  # Fixed height to keep aspect ratios intact

#             # Create a canvas with a dynamic size
#             canvas = tk.Canvas(frame, width=canvas_width, height=canvas_height, bg="white")
#             canvas.pack()

#             # Determine scaling factor for drawing
#             max_width = max(m.width for m in self.monitors)
#             max_height = max(m.height for m in self.monitors)
#             scale = min(500 / max_width, 300 / max_height)
            
#             # Arrange monitors side by side with some padding
#             padding = 20
#             x_offset = 10

#             # Draw each monitor as a rectangle
#             self.monitor_rects = []
#             for idx, monitor in enumerate(self.monitors):
#                 scaled_width = monitor.width * scale
#                 scaled_height = monitor.height * scale

#                 # Calculate position for each rectangle
#                 x_start = x_offset
#                 y_start = 200 - scaled_height // 2  # Vertically center on the canvas
#                 x_end = x_start + scaled_width
#                 y_end = y_start + scaled_height

#                 rect = canvas.create_rectangle(
#                     x_start, y_start,
#                     x_end, y_end,
#                     fill="lightblue", outline="black", tags=f"monitor_{idx}"
#                 )
#                 self.monitor_rects.append(rect)

#                 # Add monitor label
#                 canvas.create_text(
#                     (x_start + x_end) // 2, y_start - 10,
#                     text=f"Monitor {idx + 1}: {monitor.width}x{monitor.height}",
#                     fill="black"
#                 )

#                 # Make rectangles clickable
#                 canvas.tag_bind(f"monitor_{idx}", "<Button-1>", lambda e, i=idx: self.select_monitor(i))

#                 # Update x_offset for the next monitor
#                 x_offset += scaled_width + padding

#             self.canvas = canvas
#             return frame

#         def select_monitor(self, idx):
#             self.selected_monitor = idx
#             self.ok()

#         def apply(self):
#             pass

#     # Create the dialog and return the selected monitor
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window
#     dialog = MonitorSelectionDialog(root, "Select Monitor", monitors)
#     return dialog.selected_monitor


# class CocoaScreen:
#     """A class representing monitor properties."""
#     def __init__(self, x, y, width, height):
#         self.x = x
#         self.y = y
#         self.width = width
#         self.height = height

#     def __repr__(self):
#         return f"CocoaScreen(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


def restart_animation(win, RESTART_DURATION, RESTART_TEXT):
    '''
    Outer-experiment loop to restart when pressing reset
    '''
    RESTART_TEXT.draw()
    win.flip()
    core.wait(RESTART_DURATION)
