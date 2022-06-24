# -*- coding: utf-8 -*-

"""
Support function used for plotting.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Useful ancillary plotting function"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (17.11.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# plotting modules
import matplotlib.pyplot as plt
from cycler import cycler

# first make sure that modules can be loaded from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import read_csv  # read in data files


# ======================================================================

class default_plt_setup(object):

    """
    Matplotlib configuration specifics shared by all the figures

    """

    def __init__(self, colours=None, ticks=False, spines=True):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use

        # overall style
        plt.style.use('seaborn-poster')
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['axes.xmargin'] = 0.025
        plt.rcParams['axes.ymargin'] = plt.rcParams['axes.xmargin']

        # text fonts
        plt.rcParams['text.usetex'] = True  # use LaTeX
        preamble = [r'\usepackage[sfdefault,light]{roboto}',
                    r'\usepackage{sansmath}', r'\sansmath']
        plt.rcParams['text.latex.preamble'] = '\n'.join(preamble)

        # font sizes
        plt.rcParams['font.size'] = 12.
        plt.rcParams['axes.titlesize'] = plt.rcParams['font.size'] / 0.85
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size'] / 1.2
        plt.rcParams['xtick.labelsize'] = plt.rcParams['axes.labelsize']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['axes.labelsize']

        # 12 default colours
        if colours is None:  # use default
            colours = ['#600170', '#9e02b8', '#df3afb', '#ed92fd', '#f7d2fe',
                       '#024c5f', '#0f748e', '#2da2c1', '#67d2ec', '#baecff',
                       '#7b3800', '#d86302', '#ef9f08', '#ffdd11', '#fff3ab',
                       '#252525', '#636363', '#a0a0a0', '#d0d0d0']

        plt.rcParams['axes.prop_cycle'] = cycler(color=colours)
        plt.rcParams['image.cmap'] = 'bone_r'

        # lines
        plt.rcParams['lines.linewidth'] = 1.5

        # markers
        plt.rcParams['scatter.marker'] = '.'

        # boxplots
        plt.rcParams['boxplot.showcaps'] = False
        plt.rcParams['boxplot.showfliers'] = False
        plt.rcParams['boxplot.boxprops.linewidth'] = 1.5
        plt.rcParams['boxplot.whiskerprops.linewidth'] = 1.5
        plt.rcParams['boxplot.medianprops.linewidth'] = 1.5
        plt.rcParams['boxplot.medianprops.color'] = '#252525'

        # legend specs
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.columnspacing'] = 0.5
        plt.rcParams['legend.handlelength'] = 1.
        plt.rcParams['legend.markerscale'] = 0.75
        plt.rcParams['legend.fontsize'] = 10.

        # ticks
        plt.rcParams['xtick.major.size'] = plt.rcParams['font.size'] / 3.
        plt.rcParams['ytick.major.size'] = plt.rcParams['font.size'] / 4.

        # distance between ticks and ticklabels
        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['ytick.major.pad'] = plt.rcParams['xtick.major.pad']

        if not spines:
            plt.rcParams['axes.spines.left'] = False
            plt.rcParams['axes.spines.right'] = False
            plt.rcParams['axes.spines.bottom'] = False
            plt.rcParams['axes.spines.top'] = False


def amb_ele():

    """
    Sets the default "ring" plot order to use in all figures

    """

    return [['R2', 'R3', 'R6'], ['R1', 'R4', 'R5']]


def dryness(df, project):

    """
    Estimates the "dryness" of each of EucFACE's "ring" plots by
    computing their LAI x soil moisture averages

    Arguments:
    ----------
    df: pandas dataframe
        dataframe in which information on dryness needs to be added

    project: string
        path to model input files that contain information on LAI and
        observed soil moisture

    Returns:
    --------
    df: pandas dataframe
        dataframe to which information on dryness has been added

    """

    # initialise dryness column
    df['dryness'] = 0.

    files = [e for e in os.listdir(project) if e.endswith('model_drivers.csv')]

    for file in files:

        ring = file.split('_')[0].split('EucFACE')[1]
        df2, __ = read_csv(os.path.join(project, file))

        try:
            iloc = (df['ring'] == ring).index.to_list()

        except KeyError:
            iloc = (df['Ring'] == ring).index.to_list()

        df.loc[iloc, 'dryness'] = round((df2['sw'] * df2['LAI']).mean(), 6)

    return df


def correct_timeseriesticks(ax, df):

    """
    Renders timeseries x-axis ticks nicely, by only showing years

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    df: pandas dataframe
        dataframe which contains the timeseries plotted

    Returns:
    --------
    Draws timeseries ticks on the x-axis

    """

    ticks = df['Date'][df['doy'] == 1].unique().astype(np.int64)
    ticks = list(ticks) + [ticks[-1] + np.diff(ticks)[0]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(pd.DatetimeIndex(ticks).year)

    return


def render_xlabels(ax, name, unit, fs=10., pad=10.):

    """
    Renders the plotting of the x-axis label such that the unit of a
    given variable is in a smaller font, which is a nicer display

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    name: string
        name of the variable for the x-axis

    unit: string
        unit of the variable for the x-axis

    fs: float
        font size

    pad: float
        distance between the rendered label and the plot axis frame

    Returns:
    --------
    Draws the axis label on the x-axis

    """

    ax.set_xlabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.975 * fs,
                                                                unit),
                  labelpad=pad)

    return


def render_ylabels(ax, name, unit, fs=10., pad=10.):

    """
    Renders the plotting of the y-axis label such that the unit of a
    given variable is in a smaller font, which is a nicer display

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    name: string
        name of the variable for the y-axis

    unit: string
        unit of the variable for the y-axis

    fs: float
        font size

    pad: float
        distance between the rendered label and the plot axis frame

    Returns:
    --------
    Draws the axis label on the y-axis

    """

    ax.set_ylabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.975 * fs,
                                                                unit),
                  labelpad=pad)

    return
