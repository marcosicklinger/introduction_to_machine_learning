from pandas import Categorical
import numpy as np
import datetime
from matplotlib.collections import EllipseCollection
from pandas import DataFrame

def readNames(file_txt):
    ''' substitutes the date/time of creation of a tweet with a categorical variable indicate a time zone duringe the day
        Parameters
        ----------
        file_txt: file containing names of interest

        Returns
        -------
        names: list of the names contained in the given file
    '''

    # companies list
    names = []
    with open(file_txt) as file:
        while True:
            line = file.readline()
            if not line:
                break
            names.append(line.strip())
    
    return names

def plotCorrelationEllipses(data, ax = None, **kwargs):
    ''' plots correlation matrix with ellipses
        Parameters
        ----------
        data: dataframe of interest
        ax: parameter needed for plotting the matrix

        Returns
        -------
        ellipses: correlation matrix containing the correlation ellipses
    '''

    corr_matrix = np.array(data)

    if corr_matrix.ndim != 2:
        raise ValueError("Passed data 'corr_matrix' is not a matrix")
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
        ax.set_xlim([-0.5, corr_matrix.shape[0] - 0.5])
        ax.set_ylim([-0.5, corr_matrix.shape[1] - 0.5])
    
    # set coordinates for centers of ellipses
    center_coords = np.indices(corr_matrix.shape).reshape(2,-1).T

    # set ellipses' axes
    widths = np.ones_like(corr_matrix).ravel()
    heights = 1-np.abs(corr_matrix).ravel()
    angles = 45 * np.sign(corr_matrix).ravel()

    # create ellipses
    ellipses = EllipseCollection(widths=widths, heights=heights, angles=angles, units='x', offsets=center_coords, transOffset=ax.transData, array=corr_matrix.ravel(), **kwargs)

    ax.add_collection(ellipses)

    # if passed data is a dataframe use column names for graphical purposes
    if isinstance(data, DataFrame):
        ax.set_xticks(np.arange(corr_matrix.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(corr_matrix.shape[0]))
        ax.set_yticklabels(data.index)

    return ellipses