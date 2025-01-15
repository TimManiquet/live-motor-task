import subprocess
import os
import signal

#################################
# Working zone: trying to make main.py work
import sys
# from pathlib import Path

# # Add the 'src' directory to the system path
# src_path = Path(__name__).resolve().parent / 'src'
# sys.path.append(str(src_path))

# sys.path.append(".")

#################################

def launch():
    # Paths to the app and experiment scripts
    app_script = os.path.join("app", "app.py")
    experiment_script = os.path.join("experiment", "mouse_tracker_task.py")

    # Launch the app and experiment
    app_process = subprocess.Popen(["python", app_script])
    experiment_process = subprocess.Popen(["python", experiment_script])

    try:
        print("App and experiment are running. Press Ctrl+C to stop.")
        # Keep the main script alive to monitor the processes
        app_process.wait()
        experiment_process.wait()
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        # Terminate both processes gracefully
        app_process.terminate()
        experiment_process.terminate()
        # Ensure subprocesses are killed
        os.killpg(os.getpgid(app_process.pid), signal.SIGTERM)
        os.killpg(os.getpgid(experiment_process.pid), signal.SIGTERM)
        print("Processes terminated.")

if __name__ == "__main__":
    launch()