# Add the parent directory to the Python path to enable absolute imports
import os
import sys

# Get the parent directory of the src folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add it to the Python path if not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)