import os
import sys

# Define the project base directory (adjust if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the 'src' directory to sys.path for module imports
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Example paths to key directories
DATA_DIR = os.path.join(BASE_DIR, "data")
STIMULI_DIR = os.path.join(BASE_DIR, "stimuli")

# Print debug information
print(f"BASE_DIR: {BASE_DIR}")
print(f"SRC_DIR: {SRC_DIR}")
print(f"DATA_DIR: {DATA_DIR}")