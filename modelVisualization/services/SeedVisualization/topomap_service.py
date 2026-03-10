import numpy as np
from scipy.interpolate import Rbf
from modelVisualization.utils.electrode_positions import positions


def compute_topomap(values):

    pos = np.array(positions)

    x = pos[:,0]

    y = pos[:,1]

    rbf = Rbf(x,y,values,function='cubic')

    grid_x,grid_y = np.mgrid[-1:1:100j,-1:1:100j]

    grid = rbf(grid_x,grid_y)

    return grid.tolist()