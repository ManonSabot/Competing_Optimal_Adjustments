#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the temporal model simulations of Np, Vcmax, and Jmax,
and contrasts them to observations collected during various measurement
campaigns.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "plot changes in leaf nitrogen & photosynthetic capacities"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (10.01.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths
import warnings  # ignore these warnings
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# plotting modules
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # tick locators
import string  # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import amb_ele
from plot_utils import correct_timeseriesticks
from plot_utils import render_ylabels

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from analysis.analysis_utils import dirpath  # locate data
from analysis.analysis_utils import concat_rings  # concatenate dfs

# ignore these warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(project):

    """
    Main function: plot of the time-varying Vcmax and Jmax and of the
                   relative change in Np/Nleaf over time

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    Returns:
    --------
    '(project_)photocapacities.png' and '(project_)N_leaf.png' in the
    figure directory

    """

    # paths
    dir = os.path.dirname(os.path.dirname(project))
    best = os.path.join(dir, '%s_ranked.csv' % (os.path.basename(project)))

    # read in the obs files
    obs = os.path.join(os.path.join(dir.replace('output', 'input'), 'obs'),
                       'EucFACE_photo_capacity_nitrogen_dominant_trees' +
                       '_2013_2020.csv')
    obs = (pd.read_csv(obs).dropna(axis=0, how='all').dropna(axis=1, how='all')
           .squeeze())

    try:  # what is the best configuration
        best = (pd.read_csv(best).dropna(axis=0, how='all')
                .dropna(axis=1, how='all').squeeze())
        tag = best.iloc[0, 0]

        # open best files
        sims = concat_rings(project, endtag=tag, keyword='DSH')

    except FileNotFoundError:  # open default files
        sims = concat_rings(project, endtag='Rlow', keyword='DSH')

    # dates to datetime int values, necessary for plotting
    sims['Date'] = sims['Date'].values.astype(np.int64)
    obs['Date'] = (pd.to_datetime(obs['Date'], dayfirst=True).values
                   .astype(np.int64))

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project).replace('projects',
                                                           'figures'),
                          '%s_photocapacities.png' %
                          (os.path.basename(project)))

    # plot
    plot_photocapacities(obs, sims, figure)

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project).replace('projects',
                                                           'figures'),
                          '%s_N_leaf.png' % (os.path.basename(project)))

    # plot
    plot_N(obs, sims, figure)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def campaign_boxplots(ax, df, var):

    """
    Plots boxes of distributions of the observed/measured chosen
    variable, where the plots are grouped by measurement campaign.
    Average variable values for each measurement campaign are also
    added to the plot.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    df: pandas dataframe
        dataframe containing the observations to plot

    var: string
        variable to plot: 'Vmax25', 'Jmax25', or 'N'

    Returns:
    --------
    Box plots on ax.

    """

    # which zorder to overlay on
    level = max([_.zorder for _ in ax.get_children()])

    # all data points
    pos = df.groupby('Campaign').mean()['Date']
    pos = [e for e in pos if not np.all(np.isnan(np.array(e)))]
    data = df.groupby('Campaign')[var].apply(list)
    data = [np.array(e)[~np.isnan(np.array(e))] for e in data]
    ax.scatter(pos, [np.mean(e) for e in data], marker='o', s=30.,
               color=plt.rcParams['axes.prop_cycle'].by_key()['color'][-1],
               label='Average obs.', zorder=level)

    # boxplots
    magn = len(str(round(df.groupby('Campaign').mean()['Date'].iloc[0]))) - 1
    ax.boxplot(data, positions=pos, widths=0.5 * (10 ** (magn - 2.)),
               zorder=level)

    # average campaign values
    ax.scatter(df['Date'], df[var], marker='.', s=10.,
               c=plt.rcParams['boxplot.medianprops.color'], label='Obs.',
               zorder=level)

    return


def plot_photocapacities(obs, sims, fpath):

    """
    Plots the average and range of time-varying modelled Vcmax and Jmax
    across the aCO2 vs. eCO2 rings, as well as distributions of
    observations for these variables, collected during a series of
    measurement campaigns.

    Arguments:
    ----------
    obs: pandas dataframe
        dataframe containing the observations of Vcmax and Jmax

    sim: pandas dataframe
        dataframe containing the model simulations of Vcmax and Jmax

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)photocapacities.png' in the figure directory

    """

    # declare figure
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 6), sharex=True,
                             sharey='row')
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for i, var in enumerate(['Vmax25', 'Jmax25']):  # loop over vars

        # separate ambient from elevated rings
        for j, rings in enumerate(amb_ele()):

            sim = sims[sims['Ring'].isin(rings)].copy()
            sub = obs[obs['Ring'].isin(rings)].copy()

            # rename variable
            sub['Vmax25'] = sub['Vcmax25']

            # plot the simulations
            ax = axes[i * 2 + j]
            ax.fill_between(sim['Date'].unique(),
                            sim.groupby('Date').min()['%s(pmax)' % (var)],
                            sim.groupby('Date').max()['%s(pmax)' % (var)],
                            label='Range of sim.')
            next(ax._get_lines.prop_cycler)  # skip range colour
            ax.plot(sim['Date'].unique(),
                    sim.groupby('Date').mean()['%s(pmax)' % (var)],
                    label='Average sim.')

            # overlay the obs
            campaign_boxplots(ax, sub, var)

            # reference parameterisations
            ref = sim[sim['Date'] == sim['Date'].min()]
            ax.scatter([sim['Date'].min() -
                        50 * np.diff(sim['Date'].unique())[0], ] * 3,
                       [ref['%s(pmax)' % (var)].min(),
                        ref['%s(pmax)' % (var)].median(),
                        ref['%s(pmax)' % (var)].max()], marker='*', s=100.,
                       facecolor='#a777d4', edgecolor='k',
                       label='Fixed param.', zorder=15)

    for i, ax in enumerate(axes):  # format ticks, label subplots

        correct_timeseriesticks(ax, sim)
        ax.yaxis.set_major_locator(MaxNLocator(3))

        # label plots
        ax.text(0.01, 0.9, r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                transform=ax.transAxes)

    # legend
    axes[1].legend(handletextpad=0.4, ncol=3, bbox_to_anchor=(1.0125, 1.025),
                   loc=1)

    # label axes
    render_ylabels(axes[0], r'$\mathrm{V_{cmax25}}$',
                   r'µmol m$^{-2}$ s$^{-1}$',
                   fs=plt.rcParams['axes.labelsize'])
    render_ylabels(axes[2], r'$\mathrm{J_{max25}}$', r'µmol m$^{-2}$ s$^{-1}$',
                   fs=plt.rcParams['axes.labelsize'])

    # axes titles
    axes[0].set_title(r'aCO$_2$', loc='left')
    axes[1].set_title(r'eCO$_2$', loc='left')

    if not os.path.isdir(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    plt.savefig(fpath)
    plt.close()

    return


def prepare_data(obs, sim, rings, refobsN=None, refsimN=None):

    """
    Computes the change in Np or Nleaf over time, relative to the
    average Np / Nleaf at the aCO2 rings at the beginning of the
    timeseries / during the first collection campaign.

    Arguments:
    ----------
    obs: pandas dataframe
        dataframe containing the observations of Nleaf

    sim: pandas dataframe
        dataframe containing the model simulations of Np

    rings: list or array
        EucFACE's "ring" plots represented in the obs and sim data,
        either ['R2', 'R3', 'R6'] for the ambient rings or
        ['R1', 'R4', 'R5'] for the elevated CO2 rings

    refobsN: float
        reference baseline value for the relative change in Nleaf

    refsimN: float
        reference baseline value for the relative change in Np

    Returns:
    --------
    obs: pandas dataframe
        dataframe containing the relative change in observed Nleaf

    sim: pandas dataframe
        dataframe containing the relative change in simuled Np

    refobsN: float
        reference baseline value for the relative change in Nleaf

    refsimN: float
        reference baseline value for the relative change in Np

    """

    # compute total Np
    sim['N(pmax)'] = (sim['Nc(pmax)'] + sim['Ne(pmax)'] + sim['Nr(pmax)'] +
                      sim['Ns(pmax)'])

    # convert from mass basis to area basis
    obs['N.mg.g-1'] = obs['N.mg.g-1'].multiply(obs['LMA.g.m-2'])

    # change relative to the aCO2 avg
    if rings == ['R2', 'R3', 'R6']:
        ref = sim[sim['Date'] == sim['Date'].min()]
        refsimN = ref[ref['hod'] < 1.]['N(pmax)'].mean()
        ref = obs.groupby('Campaign').mean()
        refobsN = ref[ref['Date'] == ref['Date'].min()]['N.mg.g-1'].iloc[0]

    # relative changes
    sim['N(pmax)'] = 100. * (sim['N(pmax)'] - refsimN) / refsimN
    obs['N'] = 100. * (obs['N.mg.g-1'] - refobsN) / refobsN

    return obs, sim, refobsN, refsimN


def plot_N(obs, sims, fpath):

    """
    Plots the average and range of relative change in the time-varying
    modelled Np across the aCO2 vs. eCO2 rings, as well as distributions
    of the change in observed Nleaf collected during a series of
    measurement campaigns.

    Arguments:
    ----------
    obs: pandas dataframe
        dataframe containing the observations of Nleaf

    sim: pandas dataframe
        dataframe containing the model simulations of Np

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)N_leaf.png' in the figure directory

    """

    # declare figure
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.1)

    for ax in axes:  # secondary y axes to compare N

        axes = np.append(axes, ax.twinx())
        axes[-1].set_zorder(ax.get_zorder() + 1)

    # share twin axes
    axes[-2].get_shared_y_axes().join(axes[-2], axes[-1])

    # initialise the reference N values for baseline of change
    refsubN = None
    refsimN = None

    # separate ambient from elevated rings
    for i, rings in enumerate(amb_ele()):

        sim = sims[sims['Ring'].isin(rings)].copy()
        sub = obs[obs['Ring'].isin(rings)].copy()

        # rename variables, relative changes in N
        sub, sim, refsubN, refsimN = prepare_data(sub, sim, rings,
                                                  refobsN=refsubN,
                                                  refsimN=refsimN)

        # plot the simulations
        axes[i].fill_between(sim['Date'].unique(),
                             sim.groupby('Date').min()['N(pmax)'],
                             sim.groupby('Date').max()['N(pmax)'],
                             label='Range of sim.')
        next(axes[i]._get_lines.prop_cycler)  # skip range colour
        axes[i].plot(sim['Date'].unique(),
                     sim.groupby('Date').mean()['N(pmax)'],
                     label='Average sim.')

        # overlay the obs
        campaign_boxplots(axes[i + 2], sub, 'N')

    for ax in axes:  # format ticks, ticklabels

        correct_timeseriesticks(ax, sim)
        ax.yaxis.set_major_locator(MaxNLocator(3))

    # legend
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    leg = axes[0].legend(h1 + h2, l1 + l2, handletextpad=0.4, ncol=4,
                         bbox_to_anchor=(1.0125, 1.025), loc=1)
    leg.set_zorder(100)

    # label axes
    baseline = r'\overline{aCO_2}'
    render_ylabels(axes[0],
                   r'$\mathrm{\frac{\Delta N_{P}}{N_{P_{%s}}}}$' % (baseline),
                   r'$\%$', fs=1.5 * plt.rcParams['axes.labelsize'])
    render_ylabels(axes[1],
                   r'$\mathrm{\frac{\Delta N_{P}}{N_{P_{%s}}}}$' % (baseline),
                   r'$\%$', fs=1.5 * plt.rcParams['axes.labelsize'])
    render_ylabels(axes[2], r'$\mathrm{\frac{\Delta N_{leaf}}{N_{leaf_{%s}}}}$'
                   % (baseline), r'$\%$',
                   fs=1.5 * plt.rcParams['axes.labelsize'], pad=5)
    render_ylabels(axes[3], r'$\mathrm{\frac{\Delta N_{leaf}}{N_{leaf_{%s}}}}$'
                   % (baseline), r'$\%$',
                   fs=1.5 * plt.rcParams['axes.labelsize'], pad=5)

    # label plots
    axes[0].text(0.01, 0.9, r'\textbf{(%s)} %s' %
                 (string.ascii_lowercase[0], r'aCO$_2$'),
                 transform=axes[0].transAxes)
    axes[1].text(0.01, 0.9, r'\textbf{(%s)} %s' %
                 (string.ascii_lowercase[1], r'eCO$_2$'),
                 transform=axes[1].transAxes)

    if not os.path.isdir(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    plt.savefig(fpath)
    plt.close()

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
    default_plt_setup(colours=['#fff3ab', '#ef9f08', '#2da2c1'])

    # specific setup
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['lines.linewidth'] += 0.75

    # user input
    main(dirpath(args.project))
