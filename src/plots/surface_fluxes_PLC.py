#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the temporal model simulations of canopy fluxes and
hydraulic status (PLC, Pleaf), and contrasts them to observations when
possible.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "plot the canopy fluxes and canopy hydraulics"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (23.01.2022)"
__email__ = "m.e.b.sabot@gmail.com"

# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# plotting modules
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator  # tick locators
import string  # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import amb_ele
from plot_utils import correct_timeseriesticks
from plot_utils import render_ylabels

# first make sure that modules can be loaded from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM import conv  # unit converter
from analysis.analysis_utils import dirpath  # locate data
from analysis.analysis_utils import concat_rings  # concatenate dfs


# ======================================================================

def main(project, runmean, soil_hydro):

    """
    Main function: plot of the time-varying GPP, canopy E, and PLC,
                   as well as 1:1 E plots and PDFs of Pleaf

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    runmean: int
        length of the running mean to apply on the data to smooth the
        rendering and make the figure readable [days]

    soil_hydro: string
        'PSH' if plotting the model outputs that were forced with a
        prescribed soil moisture profile, 'DSH' if plotting the model
        outputs that were generated using the soil hydrological model

    Returns:
    --------
    '(project_)fluxes_PLC_DSH/PSH_Nday_smoothing.png' in the figure directory

    """

    # paths
    dir = os.path.dirname(os.path.dirname(project))
    best = os.path.join(dir, '%s_ranked.csv' % (os.path.basename(project)))

    # what is the best configuration
    best = (pd.read_csv(best).dropna(axis=0, how='all')
              .dropna(axis=1, how='all').squeeze())
    tag = best.iloc[0, 0]

    # now open drivers + default files + best files
    fins = concat_rings(project.replace('output', 'input'),
                        endtag='model_drivers')
    sims1 = concat_rings(project, endtag='Rlow', keyword=soil_hydro)
    sims2 = concat_rings(project, endtag=tag, keyword=soil_hydro)

    # dates to datetime int
    fins['Date'] = fins['Date'].values.astype(np.int64)
    sims1['Date'] = sims1['Date'].values.astype(np.int64)
    sims2['Date'] = sims2['Date'].values.astype(np.int64)

    # read in the obs files
    obs_E = os.path.join(os.path.join(dir.replace('output', 'input'), 'obs'),
                         'EucFACE_sapflow_2012_2014.csv')
    obs_E = (pd.read_csv(obs_E).dropna(axis=0, how='all')
               .dropna(axis=1, how='all').squeeze())
    obs_Psi = os.path.join(os.path.join(dir.replace('output', 'input'), 'obs'),
                           'EucFACE_water_potential_all_trees_2012_2014.csv')
    obs_Psi = (pd.read_csv(obs_Psi).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    # obs dates to datetime int
    obs_E['Date'] = (pd.to_datetime(obs_E['Date'], dayfirst=True).values
                       .astype(np.int64))
    obs_E = obs_E[obs_E['Date'] >= fins['Date'].iloc[0]]
    obs_Psi['Date'] = (pd.to_datetime(obs_Psi['Date'], dayfirst=True).values
                         .astype(np.int64))
    obs_Psi = obs_Psi[obs_Psi['Date'] >= fins['Date'].iloc[0]]

    # convert units, select relevant variables
    sims1, sims2 = prepare_data(sims1, sims2)

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project).replace('projects',
                                                           'figures'),
                          '%s_fluxes_PLC_%s_%dday_smoothing.png' %
                          (os.path.basename(project), soil_hydro, runmean))

    # plot
    plot_fluxes_PLC_insets(fins, sims1, sims2, obs_E, obs_Psi, tag, figure)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def convert_fluxes(df):

    """
    Converts the flux data to easily interpretable units.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing half-hourly simulations of canopy fluxes

    Returns:
    --------
    df: pandas dataframe
        dataframe containing half-hourly simulations of canopy fluxes

    """

    # convert units
    df[df.filter(like='An(').columns] *= (conv.umolCpm2ps_2_gCpm2phlfhr *
                                          conv.HLFHR_2_DAY)
    df[df.filter(like='E(').columns] *= (conv.mmolH2Opm2ps_2_mmphlfhr *
                                         conv.HLFHR_2_DAY)

    return df


def prepare_data(df1, df2):

    """
    Converts the flux data to easily interpretable units, estimates the
    canopy PLC from the vulnerability of sunlit and shaded components of the
    canopy.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing half-hourly simulated data from a first model
        configuration

    df2: pandas dataframe
        dataframe containing half-hourly simulated data from a second model
        configuration

    Returns:
    --------
    df1: pandas dataframe
        dataframe containing half-hourly simulated data from a first model
        configuration

    df2: pandas dataframe
        dataframe containing half-hourly simulated data from a second model
        configuration

    """

    # unit conversions
    df1 = convert_fluxes(df1)
    df2 = convert_fluxes(df2)

    # calc. PLC from the VC in sunlit & shaded components
    df1['PLC(pmax)'] = 100. * (1. - df1[['VCsun(pmax)', 'VCsha(pmax)']]
                               .mean(axis=1))
    df2['PLC(pmax)'] = 100. * (1. - df2[['VCsun(pmax)', 'VCsha(pmax)']]
                               .mean(axis=1))

    return df1, df2


def match_data(obs, sim1, sim2):

    """
    Matches overlapping dates between different dataframes and keeps
    these dates only.

    Arguments:
    ----------
    obs: pandas dataframe
        dataframe containing morning and afternoon observations of Pleaf to
        match

    sim1: pandas dataframe
        dataframe containing half-hourly simulated Pleaf data from a first
        model configuration

    sim2: pandas dataframe
        dataframe containing half-hourly simulated Pleaf data from a second
        model configuration

    Returns:
    --------
    obs: pandas dataframe
        dataframe containing matched morning and afternoon observations of
        Pleaf

    sim1: pandas dataframe
        dataframe containing matched simulated Pleaf data from a first model
        configuration

    sim2: pandas dataframe
        dataframe containing matched simulated Pleaf data from a second model
        configuration

    """

    obs = [obs[obs['Measurement'] == 'Morning'].groupby(['Date', 'Ring'])['WP']
           .mean(),
           obs[obs['Measurement'] == 'Midday'].groupby(['Date', 'Ring'])['WP']
           .mean()]
    morn = np.logical_and(sim1['hod'] >= 9., sim1['hod'] <= 12.)
    arvo = np.logical_and(sim1['hod'] >= 12.5, sim1['hod'] < 15.5)
    sim1 = [sim1[morn].groupby(['Date', 'Ring'])['Pleaf(pmax)'].mean()
            .loc[obs[0].index],
            sim1[arvo].groupby(['Date', 'Ring'])['Pleaf(pmax)'].mean()
            .loc[obs[1].index]]
    sim2 = [sim2[morn].groupby(['Date', 'Ring'])['Pleaf(pmax)'].mean()
            .loc[obs[0].index],
            sim2[arvo].groupby(['Date', 'Ring'])['Pleaf(pmax)'].mean()
            .loc[obs[1].index]]
    obs = obs[0].append(obs[1])
    sim1 = sim1[0].append(sim1[1])
    sim2 = sim2[0].append(sim2[1])

    return obs, sim1, sim2


def plot_fluxes_PLC(axes, df1, df2, label1, label2):

    """
    Plots the average and range of the time-varying GPP, E, or PLC across the
    aCO2 vs. eCO2 rings, as given by df1 and df2.

    Arguments:
    ----------
    axes: matplotlib objects
        axes on which to apply the function

    df1: pandas dataframe
        dataframe containing a first model configuration's outputs

    df2: pandas dataframe
        dataframe containing a second model configuration's ouputs

    label1: string
        label for the simulations in df1

    label2: string
        label for the simulations in df2

    Returns:
    --------
    The plotted averaged simulations and simulated ranges

    """

    for j, var in enumerate(['An', 'E', 'PLC']):

        # plot the min & max shadings at the back
        axes[2 * j].plot(df1.index, df1[('%s(pmax)' % (var), 'min')],
                         alpha=0.3, zorder=-2)
        axes[2 * j].plot(df2.index, df2[('%s(pmax)' % (var), 'min')],
                         alpha=0.3, zorder=-1)
        axes[2 * j].plot(df1.index, df1[('%s(pmax)' % (var), 'max')],
                         alpha=0.3, zorder=-2)
        axes[2 * j].plot(df2.index, df2[('%s(pmax)' % (var), 'max')],
                         alpha=0.3, zorder=-1)

        # plot the rings' average
        axes[2 * j].plot(df1.index, df1[('%s(pmax)' % (var), 'mean')], lw=2.5,
                         label=label1)
        axes[2 * j].plot(df2.index, df2[('%s(pmax)' % (var), 'mean')], lw=2.5,
                         label=label2)

    return


def plot_E2E(ax, obs, sim1, sim2):

    """
    Draws E:E 2-D histograms of the simulations to the observations, in df1 and
    then df2. Quantile regressions of E-E are also drawn above the histograms.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    obs: pandas dataframe
        dataframe containing observations of daily canopy transpiration

    sim1: pandas dataframe
        dataframe containing a first model configuration's estimates of daily
        canopy transpiration

    sim2: pandas dataframe
        dataframe containing a second model configuration's estimates of daily
        canopy transpiration

    Returns:
    --------
    The 2-D histogram plots

    """

    # 2D histogram scatter of obs to sim
    h, xedges, yedges = np.histogram2d(obs, sim1, bins=int(len(obs) / 45))
    ax.pcolormesh(xedges, yedges, h.T, norm=LogNorm(vmax=50), rasterized=True)
    h, xedges, yedges = np.histogram2d(obs, sim2, bins=int(len(obs) / 45))
    ax.pcolormesh(xedges, yedges, h.T, cmap=plt.cm.YlOrBr, alpha=0.6,
                  norm=LogNorm(vmax=50), rasterized=True)

    # Quantile regression lines above scatter
    ax.plot([np.percentile(obs, 5), np.percentile(obs, 95)],
            [np.percentile(obs, 5), np.percentile(obs, 95)], lw=2.5, color='k')
    ax.plot([np.percentile(obs, e) for e in np.arange(5, 100, 5)],
            [np.percentile(sim1, e) for e in np.arange(5, 100, 5)])
    ax.plot([np.percentile(obs, e) for e in np.arange(5, 100, 5)],
            [np.percentile(sim2, e) for e in np.arange(5, 100, 5)])

    return


def plot_Pleaf2Pleaf(ax, obs, sim1, sim2):

    """
    Draws PDFs of the simulations and the observations of morning and afternoon
    Pleaf.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    obs: pandas dataframe
        dataframe containing observations of morning and afternoon Pleaf

    sim1: pandas dataframe
        dataframe containing a first model configuration's estimates of morning
        and afternoon Pleaf

    sim2: pandas dataframe
        dataframe containing a second model configuration's estimates of
        morning and afternoon Pleaf

    Returns:
    --------
    The PDF plots

    """

    # PDFs of the obs and sims
    obs.plot.kde(ax=ax, lw=2.5, color='k')
    sim1.plot.kde(ax=ax)
    kde = sim2.plot.kde(ax=ax)

    # grabbing x and y data from the kde plot to get the modes
    obs = (kde.get_children()[0]._x)[np.argmax(kde.get_children()[0]._y)]
    sim1 = (kde.get_children()[1]._x)[np.argmax(kde.get_children()[1]._y)]
    sim2 = (kde.get_children()[2]._x)[np.argmax(kde.get_children()[2]._y)]

    # annotate the mode values
    ax.text(-0.35, 0.8,
            r'$\mathrm{\mu}$ = $\mathrm{%s}$ MPa' % (round(obs, 1)),
            size=plt.rcParams['font.size'] / 1.2, transform=ax.transAxes)
    ax.text(-0.35, 0.6,
            r'$\mathrm{\mu}$ = $\mathrm{%s}$ MPa' % (round(sim1, 1)),
            color=(plt.rcParams['axes.prop_cycle'].by_key()['color'][0]),
            size=plt.rcParams['font.size'] / 1.2, transform=ax.transAxes)
    ax.text(-0.35, 0.4,
            r'$\mathrm{\mu}$ = $\mathrm{%s}$ MPa' % (round(sim2, 1)),
            color=(plt.rcParams['axes.prop_cycle'].by_key()['color'][1]),
            size=plt.rcParams['font.size'] / 1.2, transform=ax.transAxes)

    return


def plot_fluxes_PLC_insets(fins, sims1, sims2, obs_E, obs_Psi, tag, fpath):

    """
    Plots the average and range of time-varying canopy fluxes and PLC across
    the aCO2 vs. eCO2 rings, as E:E quantile plots and PDFs of Pleaf in insets
    of the main plots.

    Arguments:
    ----------
    fins: pandas dataframe
        dataframe containing the model input data (drivers)

    sims1: pandas dataframe
        dataframe containing a first model configuration's half-hourly outputs

    sims2: pandas dataframe
        dataframe containing a second model configuration's half-hourlyouputs

    obs_E: pandas dataframe
        dataframe containing observations of daily canopy transpiration

    obs_Psi: pandas dataframe
        dataframe containing observations of morning and afternoon Pleaf

    tag: string
        characterizes the second (non-default) model configuration to plot

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)fluxes_PLC_DSH/PSH_Nday_smoothing.png' in the figure directory

    """

    # declare figure
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 9), sharex=True,
                             sharey='row')
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # add inset axis for the trans and LWPs
    bbox = axes[-1].get_window_extent()
    inaxes = [axes[i].inset_axes([1. - 0.5 * bbox.height / bbox.width,
                                  1. - 0.5, 0.5 * bbox.height / bbox.width,
                                  0.5]) for i in range(2, 6)]

    # share the magnitude x and y axes of the LWP insets
    inaxes[-2].get_shared_x_axes().join(inaxes[-2], inaxes[-1])
    inaxes[-2].get_shared_y_axes().join(inaxes[-2], inaxes[-1])

    # separate ambient from elevated rings
    for i, rings in enumerate(amb_ele()):

        obsE = obs_E[obs_E['Ring'].isin(rings)].copy()
        fin = fins[fins['Ring'].isin(rings)].copy()
        obsPsi = obs_Psi[obs_Psi['Ring'].isin(rings)].copy()
        sim1 = sims1[sims1['Ring'].isin(rings)].copy()
        sim2 = sims2[sims2['Ring'].isin(rings)].copy()

        # scale obs trans
        obsE.set_index(['Date', 'Ring'], inplace=True)
        fin.set_index(['Date', 'Ring'], inplace=True)
        obsE['volRing'] = (obsE['volRing']
                           .multiply(fin[np.isclose(fin['hod'], 12.)]
                           .loc[obsE.index, 'LAI']))

        # set Pleaf aside in order of morning, afternoon
        obsPsi, Psi1, Psi2 = match_data(obsPsi, sim1, sim2)

        # daily simulation averages per ring
        sim1 = sim1.groupby(['Date', 'Ring']).mean()
        sim2 = sim2.groupby(['Date', 'Ring']).mean()

        # set daily E aside
        E = obsE['volRing']
        E1 = sim1.loc[obsE.index, 'E(pmax)']
        E2 = sim2.loc[obsE.index, 'E(pmax)']

        # mean, min, max of obs and sim
        obsE = obsE.groupby(obsE.index.get_level_values(0)).agg(['mean', 'min',
                                                                 'max'])
        sim1 = sim1.groupby(sim1.index.get_level_values(0)).agg(['mean', 'min',
                                                                 'max'])
        sim2 = sim2.groupby(sim2.index.get_level_values(0)).agg(['mean', 'min',
                                                                 'max'])

        # plot the observations
        runmean = int(''.join([e for e in fpath if e.isdigit()]))
        axes[i + 2].plot(obsE.index,
                         obsE[('volRing', 'mean')]
                             .rolling(runmean, min_periods=1).mean(), lw=2.5,
                         color='k', label='Obs.')
        axes[i + 4].scatter(obsPsi.index.get_level_values(0),
                            [0., ] * len(obsPsi), s=40, color='k', zorder=-10)

        # plot the time-varying simulations
        plot_fluxes_PLC(axes[i:i + 5],
                        sim1.rolling(runmean, min_periods=1).mean(),
                        sim2.rolling(runmean, min_periods=1).mean(),
                        'Default',
                        r'\textit{H$_{leg}$}=%s' %
                        (''.join(filter(str.isdigit, tag.split('-')[0]))) +
                        ', ' + r'\textit{N$_{opt}$}=%s' %
                        (''.join(filter(str.isdigit, tag.split('-')[1]))))

        # inset obs E to sim E
        plot_E2E(inaxes[i], E, E1, E2)

        # inset probability density functions of Pleaf
        plot_Pleaf2Pleaf(inaxes[i + 2], obsPsi, Psi1, Psi2)

    if '_PSH_' in fpath:  # scale PLC axis to distinguish PSH runs
        bottom1, top1 = axes[-2].get_ylim()
        bottom2, top2 = axes[-1].get_ylim()
        axes[-2].set_ylim(min(bottom1, bottom2), 2. * max(top1, top2))
        axes[-1].set_ylim(min(bottom1, bottom2), 2. * max(top1, top2))

    # add legend
    axes[3].legend(handletextpad=0.4, ncol=3, bbox_to_anchor=(1.0125, 2.125),
                   loc=1)

    # share x and y ticks of the E inset axes
    inaxes[0].set_xticks([0.2, 1., 1.8])
    inaxes[0].set_yticks([0.2, 1., 1.8])
    inaxes[1].set_xticks([0.5, 1.5, 2.5])
    inaxes[1].set_yticks([0.5, 1.5, 2.5])

    for ax in inaxes[:2]:  # inset E plots, no xtick labels

        lower1, upper1 = ax.get_xlim()
        lower2, upper2 = ax.get_ylim()
        ax.set_xlim([min(lower1, lower2), max(upper1, upper2)])
        ax.set_ylim([min(lower1, lower2), max(upper1, upper2)])
        ax.tick_params(axis='both', which='major', pad=5)
        ax.set_xticklabels([])

    for ax in inaxes[2:]:  # inset Pleaf plots, no y axis

        ax.set_xlim([-5.5, 0.])
        ax.set_xticks([-4., -2.])
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis='both', which='major', pad=5)
        ax.spines['left'].set_visible(False)

    for j, ax in enumerate(axes):  # format xticks, label subplots

        correct_timeseriesticks(ax, sims1)
        ax.yaxis.set_major_locator(MaxNLocator(3))

        # label plots
        ax.text(0.01, 0.9, r'\textbf{(%s)}' % (string.ascii_lowercase[j]),
                transform=ax.transAxes)

    # label axes
    render_ylabels(axes[0], r'GPP', r'gC m$^{-2}$ d$^{-1}$',
                   fs=plt.rcParams['axes.labelsize'], pad=18.25)
    render_ylabels(axes[2], r'E', r'mm d$^{-1}$',
                   fs=plt.rcParams['axes.labelsize'], pad=10.)
    render_ylabels(axes[4], r'PLC', r'$\%$', fs=plt.rcParams['axes.labelsize'],
                   pad=12.75)

    # axes titles
    axes[0].set_title(r'$\mathrm{aCO_2}$', loc='left')
    axes[1].set_title(r'$\mathrm{eCO_2}$', loc='left')

    if not os.path.isdir(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    plt.savefig(fpath)
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = ''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-R', '--project', type=str,
                        help='folder containing the files to analyse')
    parser.add_argument('-m', '--runmean', type=int, default=90,
                        help='number of days in the running mean')
    parser.add_argument('-p', '--prescribed', action='store_true',
                        help='is the soil hydrology prescribed?')
    args = parser.parse_args()

    if args.prescribed:
        soil_hydro = 'PSH'

    else:
        soil_hydro = 'DSH'

    # default setup
    default_plt_setup(colours=['#0f748e', '#ef9f08'])

    # specific setup
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.minor.size'] = 0.
    plt.rcParams['ytick.minor.size'] = 0.

    # user input
    main(dirpath(args.project), args.runmean, soil_hydro)
