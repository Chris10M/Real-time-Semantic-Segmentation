import numpy as np


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
