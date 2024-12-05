'''
Initialise the live mouse tracker demo

Author: Tim Maniquet
Date: 2024-12-02
'''

import json

# Indicate where the external list file is located
list_file = './src/external/lists.json'

# Read the json lists from the external list file
list_data = json.load(open(list_file))

# Extract the relevant lists from the data
kwargs = list_data['kwargs']
palette = list_data['palette']