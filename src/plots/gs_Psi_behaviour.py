#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots simulations of dgs to dPleaf between the best model
configuration and the default configuration.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "plot the dgs to dPleaf between two model configurations"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (03.02.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from scipy import stats

# plotting modules
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator  # tick locators
import string  # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import amb_ele
from plot_utils import render_xlabels, render_ylabels
from plot_utils import dryness

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from analysis.analysis_utils import dirpath  # locate data
from analysis.analysis_utils import concat_rings  # concatenate dfs


# ======================================================================

def main(project):

    """
    Main function: plot of the dgs to dPleaf between the simulations of
                   two different model configurations, at each ring

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    Returns:
    --------
    '(project_)gs_Psi_relationship.png' in the figure directory

    """

    # paths
    dir = os.path.dirname(os.path.dirname(project))
    best = os.path.join(dir, '%s_ranked.csv' % (os.path.basename(project)))

    # what is the best configuration
    best = (pd.read_csv(best).dropna(axis=0, how='all')
              .dropna(axis=1, how='all').squeeze())
    tag = best.iloc[0, 0]

    # now open default files + best files
    sims1 = concat_rings(project, endtag='Rlow', keyword='DSH')
    sims2 = concat_rings(project, endtag=tag, keyword='DSH')

    # valid data only, plus get plotting order
    sims1, sims2, ring_order = prepare_data(project.replace('output', 'input'),
                                            sims1, sims2)

    # declare figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # separate ambient from elevated rings
    for i, rings in enumerate(ring_order):

        for j in range(len(rings)):

            sim1 = sims1[sims1['Ring'] == rings[j]].copy()
            sim2 = sims2[sims2['Ring'] == rings[j]].copy()

            # linear regression and scatter
            draw_2Dhistogram(axes[i + 2 * j], sim1, sim2)

            # axes labels
            axes[i + 2 * j].text(0.01, 0.9, r'\textbf{(%s)} %s' %
                                 (string.ascii_lowercase[j + i * len(rings)],
                                  rings[j]),
                                 transform=axes[i + 2 * j].transAxes)

            if j == len(rings) - 1:
                render_xlabels(axes[i + 2 * j],
                               r'$\mathrm{\Delta \Psi_{leaf}}$', r'MPa',
                               fs=plt.rcParams['axes.labelsize'])

            if i == 0:
                render_ylabels(axes[i + 2 * j], r'$\mathrm{\Delta g_{s,can}}$',
                               r'mol m$^{-2}$ s$^{-1}$',
                               fs=plt.rcParams['axes.labelsize'])

    # beautify plots
    for ax in axes:

        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlim([-0.55, 0.55])
        ax.set_ylim([-0.15, 0.15])

    # axes titles
    axes[0].set_title(r'aCO$_2$', loc='left')
    axes[1].set_title(r'eCO$_2$', loc='left')

    # save figure
    figdir = os.path.dirname(project).replace('projects', 'figures')

    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    figure = os.path.join(figdir, '%s_gs_Psi_relationship.png' %
                          (os.path.basename(project)))
    plt.savefig(figure)
    plt.close()

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def prepare_data(project, df1, df2):

    """
    Filters the data to only keep valid values of gs and Pleaf across
    the two data sets to compare. Also order the rings depending on
    dryness (i.e, average LAI x soil moisture)

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    df1: pandas dataframe
        dataframe containing the first model configuration's outputs

    df2: pandas dataframe
        dataframe containing the second model configuration's ouputs

    Returns:
    --------
    df1: pandas dataframe
        dataframe containing the first model configuration's outputs

    df2: pandas dataframe
        dataframe containing the second model configuration's ouputs

    order: nested list
        rings ordered by dryness levels in the aCO2 vs. eCO2 plots

    """

    # filter data
    mask = np.logical_and(np.logical_and(df1['gs(pmax)'] >= 0.,
                                         df2['gs(pmax)'] >= 0.),
                          np.logical_and(df1['Pleaf(pmax)'] < df1['Ps(pmax)'],
                                         df2['Pleaf(pmax)'] < df2['Ps(pmax)']))

    df1 = df1[mask].copy()
    df2 = df2[mask].copy()

    # dryness info by ring to figure out order of plot
    df1 = dryness(df1, project)
    order = df1.groupby(['Ring'])['dryness'].mean().sort_values().index
    order = [order[order.isin(amb_ele()[0])].to_list(),
             order[order.isin(amb_ele()[1])].to_list()]

    return df1, df2, order


def draw_2Dhistogram(ax, df1, df2):

    """
    Draws 2-D histograms of the differences between the gs of df1 and
    df2 to differences between the Pleaf of df1 and df2. A linear
    regression of dgs-dPleaf is also drawn behind the histograms.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    df1: pandas dataframe
        dataframe containing the first model configuration's outputs

    df2: pandas dataframe
        dataframe containing the second model configuration's ouputs

    Returns:
    --------
    The 2-D histogram plots

    """

    # linear regression line behind the scatter
    sl, inter, __, __, __ = stats.linregress(df2['Pleaf(pmax)'] -
                                             df1['Pleaf(pmax)'],
                                             df2['gs(pmax)'] - df1['gs(pmax)'])
    r, __ = stats.spearmanr(df2['Pleaf(pmax)'] - df1['Pleaf(pmax)'],
                            df2['gs(pmax)'] - df1['gs(pmax)'])
    ax.plot(df2['Pleaf(pmax)'] - df1['Pleaf(pmax)'],
            inter + sl * (df2['Pleaf(pmax)'] - df1['Pleaf(pmax)']), zorder=-10)
    ax.text(0.975, 0.05, r'slope = %.2f' % (round(sl, 2)) +
            '\n' + r'$\rho$ = %.2f' % (round(r, 2)), ha='right',
            transform=ax.transAxes)

    # define histogram
    h, xedges, yedges = np.histogram2d(df2['Pleaf(pmax)'] - df1['Pleaf(pmax)'],
                                       df2['gs(pmax)'] - df1['gs(pmax)'],
                                       bins=500)
    ax.pcolormesh(xedges, yedges, h.T, norm=LogNorm(vmax=150),
                  rasterized=True)

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-R', '--project', type=str,
                        help='folder containing the files to analyse')
    args = parser.parse_args()

    # default setup
    default_plt_setup(colours=['#ef9f08'])

    # specific setup
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size'] + 2.

    # user input
    main(dirpath(args.project))
