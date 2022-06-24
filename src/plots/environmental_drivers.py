#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots key model drivers.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "plot the model drivers"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (20.01.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths

# plotting modules
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # tick locators

# own modules
from plot_utils import default_plt_setup
from plot_utils import render_ylabels

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from analysis.analysis_utils import dirpath  # locate data
from analysis.analysis_utils import concat_rings  # concatenate dfs


# ======================================================================

def main(project):

    """
    Main function: plot of the observed PPFD, Tair, VPD, precip, and LAI

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    Returns:
    --------
    '(project_)environmental_drivers.png' in the figure directory

    """

    # read in the drivers' files
    fins = concat_rings(project.replace('output', 'input'),
                        endtag='model_drivers')

    # rings as indices
    fins.set_index('Ring', inplace=True)

    # declare figure
    fig, axes = plt.subplots(5, 1, figsize=(6, 8), sharex=True)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.1)

    # drivers are the same across rings, except for LAI
    default = fins.loc['R1']
    axes[0].scatter(default['Date'][default['PPFD'] > 0.],
                    default['PPFD'][default['PPFD'] > 0.], s=1, alpha=0.2)
    axes[1].scatter(default['Date'], default['Tair'], s=1, alpha=0.2)
    axes[2].scatter(default['Date'], default['VPD'], s=1, alpha=0.2)

    # express weekly for the precip
    default = default.groupby('Date').mean()  # mm d-1
    default = default.resample('M').sum()  # mm m-1
    axes[3].step(default.index, default['precip'])

    # now loop over rings for LAI
    next(axes[4]._get_lines.prop_cycler)  # skip grey

    for ring in ['R2', 'R3', 'R6', 'R1', 'R4', 'R5']:

        LAI = fins.loc[ring]
        axes[4].plot(LAI['Date'], LAI['LAI'], label=ring)

    axes[4].legend(handletextpad=0.4, ncol=2, bbox_to_anchor=(1.0125, 1.05),
                   loc=1)

    for ax in axes:  # format y axis

        ax.yaxis.set_major_locator(MaxNLocator(3))

    # axes labels
    render_ylabels(axes[0], r'$\mathrm{PPFD}$',
                   r'$\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$',
                   fs=plt.rcParams['axes.labelsize'], pad=5)
    render_ylabels(axes[1], r'$T_{a}$', r'$^\circ$C',
                   fs=plt.rcParams['axes.labelsize'], pad=15.75)
    render_ylabels(axes[2], r'$D_{a}$', r'kPa',
                   fs=plt.rcParams['axes.labelsize'], pad=21.25)
    render_ylabels(axes[3], r'$\mathrm{precip}$', r'mm mon$^{-1}$',
                   fs=plt.rcParams['axes.labelsize'], pad=9.5)
    render_ylabels(axes[4], r'$\mathrm{LAI}$', r'm$^2$ m$^{-2}$',
                   fs=plt.rcParams['axes.labelsize'], pad=11.25)

    # save figure
    figdir = os.path.dirname(project).replace('projects', 'figures')

    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    figure = os.path.join(figdir, '%s_environmental_drivers.png' %
                          (os.path.basename(project)))
    plt.savefig(figure)
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = ''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-R', '--project', type=str,
                        help='folder containing the files to analyse')
    args = parser.parse_args()

    # default setup
    default_plt_setup(colours=['#a0a0a0', '#d86302', '#ef9f08', '#ffdd11',
                               '#7311d0', '#9d27e6', '#a777d4'])

    # specific setup
    plt.rcParams['legend.fontsize'] -= 2.

    # user input
    main(dirpath(args.project))
