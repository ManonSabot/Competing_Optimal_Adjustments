#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots differences in PLC induced by changes in LAI, and
associated changes in C allocation to leaves.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Jiang, M., Medlyn, B. E., Drake, J. E., Duursma, R. A., Anderson, I.
  C., Barton, C. V., ... & Ellsworth, D. S. (2020). The fate of carbon
  in a mature forest under carbon dioxide enrichment. Nature, 580(7802),
  227-231.

"""

__title__ = "plot PLC-LAI-biomass tradeoffs"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (03.05.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator  # tick locators
import string  # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import amb_ele
from plot_utils import correct_timeseriesticks
from plot_utils import render_ylabels

from surface_fluxes_PLC import prepare_data

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM import conv  # unit converter
from analysis.analysis_utils import dirpath  # locate data
from analysis.analysis_utils import concat_rings  # concatenate dfs


# ======================================================================

def main(project1, project2):

    """
    Main function: plot of differences in PLC associated with changes in
                   LAI, as well as the coinciding C allocation to leaves

    Arguments:
    ----------
    project1: string
        project repository where the files containing the data to plot
        for a given LAI are (with path)

    project2: string
        project repository where the files containing the data to plot
        for a different LAI are (with path)

    Returns:
    --------
    '(project_)LAI_v_PLC.png' and '(project_)PLC_LAI_tradeoff.png' in
    the figure directory

    """

    # what is the best configuration?
    tags = [e.split('Rlow-')[1].split('.csv')[0] for e in os.listdir(project1)]

    # open all necessary the files
    fins = concat_rings(project1.replace('output', 'input'),
                        endtag='model_drivers')
    rfins = concat_rings(project2.replace('output', 'input'),
                         endtag='model_drivers')
    sims = concat_rings(project1, keyword='DSH')
    rsims = concat_rings(project2, endtag=tags[0], keyword='DSH')

    # C budget parameter file
    fname = os.path.join(os.path.dirname(os.path.dirname(project1
                         .replace('output', 'input'))),
                         'C_budget_model_params.csv')
    Cparams = pd.read_csv(fname)

    # dates to datetime int, dates and rings as indices
    fins = format_dataset(fins)
    rfins = format_dataset(rfins)
    sims = format_dataset(sims)
    rsims = format_dataset(rsims)

    # convert units, select relevant variables
    sims, rsims = prepare_data(sims, rsims)

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project2).replace('projects',
                                                            'figures'),
                          '%s_LAI_v_PLC_v_Calloc.png' %
                          (os.path.basename(project2)))

    # plot
    plot_LAI_PLC_biomass_coincidence(fins, rfins, sims, rsims, Cparams, figure)

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project2).replace('projects',
                                                            'figures'),
                          '%s_PLC_LAI_tradeoff.png' %
                          (os.path.basename(project2)))

    # plot
    plot_delta_LAI_PLC_tradeoff(fins, rfins, sims, rsims, figure)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def format_dataset(df):

    """
    Converts the dataset dates to int and removes objects types for
    quicker data manipulation.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe to format

    Returns:
    --------
    df: pandas dataframe
        formatted dataframe

    """

    # dates to datetime int, necessary for plotting
    df['Date'] = df['Date'].values.astype(np.int64)

    # dates and rings as indices
    df.set_index(['Date', 'Ring'], inplace=True)

    # exclude strings from dfs
    df = df.select_dtypes(exclude=['object'])

    return df


def C_budget(GPP, C):

    """
    Allocates carbon to different pools based on a simple C budget model
    specific to EucFACE (see Extended Data figure 7 in Jiang et al.,
    2020). Turnover is ignored here, as we're interested in the
    cumulative C allocated at the end of the hydrological year, and we
    restart the C budget at zero for each new year.

    Arguments:
    ----------
    GPP: pandas series
        model estimates of each ring's GPP

    C: pandas dataframe
        dataframe containing C allocation and turnover coefficients for
        EucFACE's aCO2 vs. eCO2 rings

    Returns:
    --------
    Cpools: pandas dataframe
        dataframe containing the cumulative allocation of C into
        different pools for each hydrological year considered

    """

    # index the C budget parameters per ring to match the GPP data
    C.set_index('Ring', inplace=True)

    # set up C component pools in a df
    Cpools = pd.DataFrame(columns=['Cleaf', 'Cwood', 'Croot', 'Cmyco'])

    # make 1st July (hydro year) the 1st doy
    dates = (pd.to_datetime(GPP.index.get_level_values(0)) -
             pd.DateOffset(months=6)).values.astype(np.int64)
    GPP.index = pd.MultiIndex.from_tuples(list(zip(dates,
                                          GPP.index.get_level_values(1))),
                                          names=['Date', 'Ring'])

    # the C budget is applied per year
    years = pd.DatetimeIndex(GPP.index.levels[0]).year.unique()

    for y in years:

        # year GPP
        sub = (GPP.loc[pd.DatetimeIndex(GPP.index.get_level_values(0)).year ==
                       y] / conv.HLFHR_2_DAY)

        # average respiration for the year
        Resp = ((1. - C.CUE) * sub).groupby(level=1).mean()

        # allocate NPP
        Cleaf = sub.sub(Resp, level=1).multiply(C.leaf_alloc, level=1)
        Cwood = sub.sub(Resp, level=1).multiply(C.wood_alloc, level=1)
        Croot = sub.sub(Resp, level=1).multiply(C.root_alloc, level=1)
        Cmyco = sub.sub(Resp, level=1).multiply(C.myco_alloc, level=1)

        # cumulative amounts
        Cleaf = Cleaf.groupby(level=1).cumsum()
        Cwood = Cwood.groupby(level=1).cumsum()
        Croot = Croot.groupby(level=1).cumsum()
        Cmyco = Cmyco.groupby(level=1).cumsum()

        # put each year's values in df
        Cpools = Cpools.append(pd.DataFrame({'Cleaf': Cleaf, 'Cwood': Cwood,
                                             'Croot': Croot, 'Cmyco': Cmyco}))

    Cpools.index = pd.MultiIndex.from_tuples(Cpools.index)

    # now back to calendar years
    dates = (pd.to_datetime(Cpools.index.get_level_values(0)) +
             pd.DateOffset(months=6)).values.astype(np.int64)
    Cpools.index = pd.MultiIndex.from_tuples(list(zip(dates,
                                             (Cpools.index
                                                    .get_level_values(1)))),
                                             names=['Date', 'Ring'])

    return Cpools


def year2year_relative(df1, df2):

    """
    Computes the net end of (hydrological) year difference in C storage
    between df1 and df2.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing the cumulative allocation of C into
        different pools for each hydrological year considered

    df2: pandas dataframe
        dataframe containing alternative estimates of the cumulative
        allocation of C into different pools for each hydrological year
        considered

    Returns:
    --------
    df1: pandas dataframe
        dataframe containing the cumulative allocation of C into
        different pools for each hydrological year considered

    df2: pandas dataframe
        dataframe containing alternative estimates of the cumulative
        allocation of C into different pools for each hydrological year
        considered

    excess: pandas dataframe
        dataframe containing the net excess end of (hydrological) year C
        in each C pool in df1 compared to df2

    """

    # add to each new year: diff from max year ante
    years = pd.to_datetime(df1.index.levels[0]).year.unique()
    idx = (pd.to_datetime(pd.DataFrame({'year': list(years),
                                        'month': [6, ] * len(years),
                                        'day': [30, ] * len(years)}))
           .values.astype(np.int64))  # end of hydro year
    when = df1.loc[idx].groupby(df1.loc[idx].index).max()
    when.index = pd.MultiIndex.from_tuples(when.index, names=['Date', 'Ring'])

    # store excess
    excess = df1.copy()
    excess.where(excess.isna(), 0., inplace=True)

    for i in range(len(when.index.levels[0])):

        i1 = when.index.levels[0][i]

        try:
            i2 = when.index.levels[0][i + 1]
            idx = df1.index.levels[0][np.argwhere(df1.index.levels[0] == i1)
                                      [0][0] + 1:
                                      np.argwhere(df1.index.levels[0] == i2)
                                      [0][0] + 1].to_list()

        except IndexError:
            idx = df1.index.levels[0][np.argwhere(df1.index.levels[0] == i1)
                                      [0][0] + 1:].to_list()

        max1 = df1.loc[i1].groupby(df1.loc[i1].index).max()
        max2 = df2.loc[i1].groupby(df2.loc[i1].index).max()
        excess.loc[idx] = excess.loc[idx].sub(max1.sub(max2), axis=1)

    return df1, df2, excess


def plot_carry_over_Cleaf(ax, exc):

    """
    Plots a bar chart of the net excess end of (hydrological) year C in
    each C pool in one dataset compared to another.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    exc: pandas dataframe
        dataframe containing the net excess end of (hydrological) year C
        in each C pool in one dataset compared to another

    Returns:
    --------
    Bar plot insets in ax.

    """

    # inset axis
    ax = ax.inset_axes([0.7, 0.7, 0.3, 0.3])

    # add end of year excess
    exc = exc.groupby(pd.DatetimeIndex(exc.index).year).mean().cumsum()

    # bar plot
    ax.bar(np.arange(len(exc)), exc['Cleaf'], color='#748891')

    # how much excess is there at most
    ax.annotate(round(exc['Cleaf'].max()),
                xy=(np.arange(len(exc))[np.argmax(exc['Cleaf'])],
                exc['Cleaf'].max()), xytext=(0.5, -5.), xycoords='data',
                textcoords='offset points', va='center', ha='center', c='w',
                size=7)

    # format axis ticks
    ax.set_xticks(np.arange(len(exc))[::2])
    ax.set_xticklabels(exc.index[::2], rotation=55)
    ax.tick_params(length=2., width=1., pad=1, labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(2))

    # label axis
    ax.set_ylabel(r'$\mathrm{\Delta_{y} A_{fcum}}$', size=8, labelpad=4)

    return


def PLC_colourbar(ax, norm):

    """
    Plots a colourbar arrow that characterises the 'background' PLC on
    which PLC risk is added.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    norm: matplotlib object
        how to map the colours in the colourmap, from data values vmin
        to vmax

    Returns:
    --------
    Coloured arrow adjacent to ax.

    """

    # axis for colourbar
    cax = ax.inset_axes([1.03, 0.1, 0.035, 0.825])

    # draw colourbar
    cb = mpl.colorbar.ColorbarBase(cax, norm=norm,
                                   ticks=[10, 50, 90], extend='max',
                                   extendfrac=0.1)

    # format colourbar ticks
    cb.set_ticklabels([r'10$\%$', r'50$\%$', r'90$\%$'])
    cax.tick_params(length=0., width=0., pad=2, labelsize=8)
    cb.outline.set_visible(False)
    cax.set_ylim([0, 100])

    # label colourbar
    cax.set_title(r'$\mathrm{PLC_{\overline{LAI}}}$', y=0.978, size=9)

    return


def plot_LAI_PLC_biomass_coincidence(fins1, fins2, sims1, sims2, Cparams,
                                     fpath):

    """
    Plots the difference in PLC associated with a change in LAI between
    two sets of model simulations at EucFACE's aCO2 vs. eCO2 rings, as
    well as the associated C allocation to leaves in each set of
    simulations.

    Arguments:
    ----------
    fins1: pandas dataframe
        dataframe containing a first set of model input data (drivers)

    fins2: pandas dataframe
        dataframe containing an alternative set of model input data
        (drivers), i.e., alternative LAI

    sims1: pandas dataframe
        dataframe containing model simulation outputs generated using
        the data from fins1 as drivers

    sims2: pandas dataframe
        dataframe containing model simulation outputs generated using
        the data from fins2 as drivers

    Cparams: pandas dataframe
        dataframe containing C allocation and turnover coefficients for
        EucFACE's aCO2 vs. eCO2 rings

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)LAI_v_PLC_v_Calloc.png' in the figure directory

    """

    # differences between the two runs
    drelLAI = 100. * (fins2.sub(fins1)).divide(fins1)  # relative diff
    dPLC = sims2.sub(sims1)  # delta
    dPLC.where(dPLC['PLC(pmax)'] != 0., inplace=True)  # hide zeros

    # associated C budget
    Cpools, rCpools, excess = year2year_relative(C_budget(sims1['An(pmax)'],
                                                          Cparams.copy()),
                                                 C_budget(sims2['An(pmax)'],
                                                          Cparams.copy()))

    # declare figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True, sharey='row')
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # use sims PLC to colour the PLC risk
    dPLC['c'] = sims1['PLC(pmax)']
    norm = Normalize(vmin=dPLC['c'][dPLC['c'] > 0.].min() - 10.,
                     vmax=dPLC['c'].max())  # no whites

    # separate ambient from elevated rings
    for i in range(len(amb_ele())):

        for j in range(len(amb_ele()[i])):

            # plot each ring's LAI separately
            LAI = drelLAI.loc[drelLAI.index.get_level_values(1) ==
                              amb_ele()[i][j]]
            colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

            if j == 0:
                axes[i].fill_between(LAI.index.get_level_values(0), LAI['LAI'],
                                     alpha=0.5, color=colour,
                                     label=r'$\mathrm{\Delta_{rel} LAI}$')

            else:
                axes[i].fill_between(LAI.index.get_level_values(0), LAI['LAI'],
                                     alpha=0.5, color=colour)

        # plot all rings' PLCs together
        PLC = dPLC.loc[dPLC.index.get_level_values(1).isin(amb_ele()[i])]
        axes[i].scatter(PLC.index.get_level_values(0), PLC['PLC(pmax)'],
                        marker='.', s=5, c=PLC['c'], alpha=1. / 3., norm=norm)

        # C alloc model param are not ring-specific, avg amb vs ele
        colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][-2]
        rC = (rCpools.loc[rCpools.index.get_level_values(1).isin(amb_ele()[i])]
              .groupby(level=0).mean())
        axes[i + 2].fill_between(rC.index, rC['Cleaf'], alpha=0.3, ec='k',
                                 color=colour, label=r'$\mathrm{LAI}$')

        # same thing for average LAI runs
        colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][-1]
        C = (Cpools.loc[Cpools.index.get_level_values(1).isin(amb_ele()[i])]
             .groupby(level=0).mean())
        axes[i + 2].fill_between(C.index, C['Cleaf'], alpha=0.3, ec='k',
                                 color=colour,
                                 label=r'$\mathrm{\overline{LAI}}$')

        # inset excess eoy Cleaf
        exc = (excess.loc[excess.index.get_level_values(1).isin(amb_ele()[i])]
               .groupby(level=0).mean())
        plot_carry_over_Cleaf(axes[i + len(amb_ele())], exc)

    # annotate LAI - PLC plot
    axes[0].annotate('Higher LAI, extra PLC', xy=(0.25, 0.78),
                     xytext=(0.3, 0.92), xycoords='axes fraction', va='center',
                     ha='left', size=9,
                     arrowprops=dict(arrowstyle='-|>', fc='k'))
    axes[0].annotate('Lower LAI, avoided PLC', xy=(0.775, 0.35),
                     xytext=(0.65, 0.08), xycoords='axes fraction',
                     va='center', ha='center', size=9,
                     arrowprops=dict(arrowstyle='-|>', fc='k'))

    # add PLC 'added risk' colourbar
    PLC_colourbar(axes[1], norm)

    # add legend
    axes[1].legend(handletextpad=0.4, bbox_to_anchor=(1.008, -0.05), loc=3)
    axes[-1].legend(handletextpad=0.4, bbox_to_anchor=(1.008, -0.05), loc=3)

    # tighten bottom subplots
    __, top1 = axes[-2].get_ylim()
    __, top2 = axes[-1].get_ylim()
    axes[-2].set_ylim(0., 1.15 * max(top1, top2))
    axes[-1].set_ylim(0., 1.15 * max(top1, top2))

    # the date variable has been removed
    sims1['Date'] = sims1.index.get_level_values(0)

    for j, ax in enumerate(axes):  # format ticks, label subplots

        correct_timeseriesticks(ax, sims1)
        ax.yaxis.set_major_locator(MaxNLocator(3))

        # label plots
        ax.text(0.01, 0.9, r'\textbf{(%s)}' % (string.ascii_lowercase[j]),
                transform=ax.transAxes)

    # label axes
    v1 = (r'{\fontsize{%spt}{3em}\selectfont{}$\mathrm{\Delta PLC}$ }' %
          (plt.rcParams['axes.labelsize']))
    v2 = (r'{\fontsize{%spt}{3em}\selectfont{}$\mathrm{\Delta_{rel} LAI}$ }' %
          (plt.rcParams['axes.labelsize']))
    axes[0].set_ylabel(v1 + r'($\%$) $\vert$ ' + v2 + r'($\%$)', labelpad=8)
    render_ylabels(axes[2], r'$\mathrm{A_{fcum}}$', r'gC m$^{-2}$',
                   fs=plt.rcParams['axes.labelsize'])

    # axes titles
    axes[0].set_title(r'aCO$_2$', loc='left')
    axes[1].set_title(r'eCO$_2$', loc='left')

    if not os.path.isdir(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    plt.savefig(fpath)
    plt.close()

    return


def plot_delta_LAI_PLC_tradeoff(fins1, fins2, sims1, sims2, fpath):

    """
    Plots the tradeoff between a change in PLC and an associated change
    in LAI (i.e., dPLC / dLAI) overtime at EucFACE's aCO2 vs. eCO2
    rings, for notable LAI changes (>0.2 m2 m-2).

    Arguments:
    ----------
    fins1: pandas dataframe
        dataframe containing a first set of model input data (drivers)

    fins2: pandas dataframe
        dataframe containing an alternative set of model input data
        (drivers), i.e., alternative LAI

    sims1: pandas dataframe
        dataframe containing model simulation outputs generated using
        the data from fins1 as drivers

    sims2: pandas dataframe
        dataframe containing model simulation outputs generated using
        the data from fins2 as drivers

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)PLC_LAI_tradeoff.png' in the figure directory

    """

    # differences between the two runs
    dLAI = fins2.sub(fins1)  # delta
    dLAI['LAI'].where(np.logical_or(dLAI['LAI'] <= -0.2, dLAI['LAI'] >= 0.2),
                      inplace=True)  # noticeable delta
    dPLC = sims2.sub(sims1)  # delta
    dPLC.where(dPLC['PLC(pmax)'] != 0., inplace=True)  # hide zeros

    # tradeoff in deltas
    deltadelta = dPLC['PLC(pmax)'].divide(dLAI['LAI'])

    # declare figure
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.1)

    # separate ambient from elevated rings
    for i in range(len(amb_ele())):

        maximums = []

        for j in range(len(amb_ele()[i])):

            delta = deltadelta.loc[deltadelta.index.get_level_values(1) ==
                                   amb_ele()[i][j]]
            LAI = dLAI.loc[dLAI.index.get_level_values(1) == amb_ele()[i][j],
                           'LAI']

            # colour and label for LAI > 0
            colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
            label = r'$\mathrm{\Delta LAI\/\geq\/0.2}$' + \
                    r' $\mathrm{m^{2}m^{-2}}$'

            for dlt in [delta.where(LAI > 0.), delta.where(LAI < 0.)]:

                if j == 0:
                    axes[i].fill_between(dlt.index.get_level_values(0), dlt,
                                         alpha=0.5, color=colour, label=label)

                else:
                    axes[i].fill_between(dlt.index.get_level_values(0), dlt,
                                         alpha=0.5, color=colour)

                maximums += [(dlt.index.get_level_values(0)[dlt.argmax()],
                              dlt.max())]

                # colour and label for LAI < 0
                colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
                label = r'$\mathrm{\Delta LAI\/\leq\/-0.2}$' + \
                        r' $\mathrm{m^{2}m^{-2}}$'

        # mark the max
        posLAI = maximums[::2]
        negLAI = maximums[1::2]
        ipos = [e[1] for e in posLAI].index(max([e[1] for e in posLAI]))
        ineg = [e[1] for e in negLAI].index(max([e[1] for e in negLAI]))

        # annotate plot
        axes[i].annotate(str(round(posLAI[ipos][1])),
                         xy=(posLAI[ipos][0], posLAI[ipos][1]),
                         xytext=(30, 10), textcoords='offset points',
                         va='center', ha='left', size=9,
                         arrowprops=dict(arrowstyle='-|>', fc='k'))
        axes[i].annotate(str(round(negLAI[ineg][1])),
                         xy=(negLAI[ineg][0], negLAI[ineg][1]),
                         xytext=(-35, -15), textcoords='offset points',
                         va='center', ha='left', size=9,
                         arrowprops=dict(arrowstyle='-|>', fc='k'))

    # add legend
    axes[0].legend(loc=1)

    # the date variable has been removed
    sims1['Date'] = sims1.index.get_level_values(0)

    for j, ax in enumerate(axes):  # format ticks, label subplots

        correct_timeseriesticks(ax, sims1)
        ax.yaxis.set_major_locator(MaxNLocator(3))

        # label axes
        ax.set_ylabel(r'$\mathrm{\frac{\Delta PLC}{\Delta LAI}}$',
                      size=plt.rcParams['axes.labelsize'] + 4., labelpad=5)

    # label plots
    axes[0].text(0.01, 0.9, r'\textbf{(%s)} %s' % (string.ascii_lowercase[0],
                                                   r'aCO$_2$'),
                 transform=axes[0].transAxes)
    axes[1].text(0.01, 0.9, r'\textbf{(%s)} %s' % (string.ascii_lowercase[1],
                                                   r'eCO$_2$'),
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
    parser.add_argument('project1', type=str,
                        help='folder containing files to analyse')
    parser.add_argument('project2', type=str,
                        help='folder containing files to analyse')
    args = parser.parse_args()

    # default setup
    default_plt_setup(colours=['#a777d4', '#ef9f08', '#024c5f', '#67d2ec'])

    # specific setup
    plt.rcParams['image.cmap'] = 'YlOrBr'
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] -= 1.
    plt.rcParams['legend.handlelength'] += 0.3
    plt.rcParams['legend.handleheight'] += 0.1

    # user input
    main(dirpath(args.project1), dirpath(args.project2))
