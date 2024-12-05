"""
Generate random colour images to run a mock experiment.

author Tim Maniquet
created 4 November 2024
"""

# The code below was used to generate images

import numpy as np
import matplotlib.pyplot as plt

n_stimuli = 10

size = 128

for i in range(n_stimuli):
    ch1 = np.ones((size, size)) * np.random.random()
    ch2 = np.ones((size, size)) * np.random.random()
    ch3 = np.ones((size, size)) * np.random.random()
    image = np.stack([ch1, ch2, ch3], axis = -1)
    plt.imsave(f'stimuli/image_{i}.png', image)


for i in range(n_stimuli):
    ch1 = np.ones((size, size)) * np.random.random()
    ch2 = np.ones((size, size)) * np.random.random()
    ch3 = np.ones((size, size)) * np.random.random()
    image = np.stack([ch1, ch2, ch3], axis = -1)
    plt.imsave(f'stimuli/negative_{i}.png', image)

