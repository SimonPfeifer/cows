import cows
import numpy as np

ncells = 10

# Set up some example data
data = np.zeros([ncells,ncells,ncells])
data[1:9, 3:8, 3:8] = 1
data[3:8, 1:9, 3:8] = 1
# data[3:8, 3:8, 1:9] = 1

# Generate the skeleton
skeleton = cows.skeletonize(data)

# Separate the skeleton
filaments = cows.separate_skeleton(skeleton)

# Generate the filament catalogue
catalogue = cows.gen_catalogue(filaments)

print(catalogue)
