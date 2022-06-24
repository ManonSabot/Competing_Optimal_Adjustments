#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the model configurations' ability to match
observations of transpiration, as defined by statistical metrics of
performance/skill.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "plot the performance scores of the various configurations"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (03.05.2022)"
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
from matplotlib.ticker import MaxNLocator  # tick locators
import matplotlib.patches as mpatches  # custom legend

# own modules
from plot_utils import default_plt_setup
from plot_utils import amb_ele

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from analysis.analysis_utils import dirpath  # locate data


# ======================================================================

def main(project):

    """
    Main function: plot of statistical metrics of performance/skill for
                   the different model configurations

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    Returns:
    --------
    '(project_)performance.png' in the figure directory

    """

    # read in the perf files
    perf = os.path.join(os.path.dirname(os.path.dirname(project)),
                        '%s_E_perf.csv' % (os.path.basename(project)))
    df1 = (pd.read_csv(perf).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())
    rank = os.path.join(os.path.dirname(os.path.dirname(project)),
                        '%s_ranked.csv' % (os.path.basename(project)))
    df2 = (pd.read_csv(rank).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

    # keep the data for the ProfitMax model only
    df1 = df1[df1['model'] == 'pmax']

    # prepare data for plotting
    metrics = ['variability', 'similarity', 'accuracy']
    signs = [-1, -1, 1]
    df1 = prepare_data(df1, metrics, signs)

    # groups of tags
    Legs = [['Leg14'], ['Leg30'], ['Leg60', 'Leg90'], ['Leg180']]
    JVs = [['JV7', 'JV14'], ['JV21'], ['JV28', 'JV35']]
    df1, cdic = assign_colours(df1, Legs, JVs)

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project).replace('projects',
                                                           'figures'),
                          '%s_performance.png' % (os.path.basename(project)))

    # plot
    snake_plots(df1, df2, metrics, signs, Legs, JVs, cdic, figure)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def prepare_data(df, metrics, signs):

    """
    Filters the data to only keep the performance metrics for the ambient
    ring's transpiration estimates, and modify the metrics so they are all
    ordered from best to worst performance.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing information on the different model configurations'
        performance / skill as given by a number of statistical metrics

    metrics: list
        statistical metrics to consider

    signs: list
        positive if the lowest skill score indicates the highest performance,
        negative otherwise

    Returns:
    --------
    df: pandas dataframe
        dataframe containing information on the different model configurations'
        performance / skill as given by some chosen statistical metrics

    """

    # only keep ambient rings
    df = df[df['ring'].isin(amb_ele()[0])]

    # better, simpler tags
    df = (df.replace({'tag': 'P-DSH-F-Rlow'}, {'tag': 'default'}, regex=True)
            .replace({'tag': 'default-'}, {'tag': ''}, regex=True))
    df = df.groupby('tag').mean()

    # order metrics from best to worst performance in all cases
    for i in range(len(signs)):

        if signs[i] < 0:  # alter the metric for it to decrease
            df[metrics[i]] = 1. - df[metrics[i]]

            if any(df[metrics[i]]) < 0.:  # e.g., variability > 1
                df[metrics[i]] = df[metrics[i]].abs()

    return df


def assign_colours(df, Legs, JVs):

    """
    Assigns different colours to different model configurations (i.e., pairwise
    combinations of elements from Legs and JVs).

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing information on the different model configurations'
        performance / skill as given by some chosen statistical metrics

    Legs: nested list
        each sublist consists in an individual hydraulic legacies element to
        consider for the paiwise combinations of Hleg and Nopt

    JVs: nested list
        each sublist consists in an individual nitrogen optimisation element
        to consider for the paiwise combinations of Hleg and Nopt

    Returns:
    --------
    df: pandas dataframe
        dataframe containing information on the different model configurations'
        performance / skill as given by some chosen statistical metrics

    cdic: dictionary
        colours associated with each pairwise combination of Hleg and Nopt

    """

    # pairings of tags by groups
    group1 = [['%s-%s' % (e, JVs[0][0]), '%s-%s' % (e, JVs[0][1])] for e in
              sum(Legs, [])] + [JVs[0]]
    group1 = [group1[0], group1[1], sum(group1[2:4], []), group1[4],
              group1[-1]]
    group2 = [['%s-%s' % (e, JVs[1][0])] for e in
              sum(Legs, [])] + [JVs[1]]
    group2 = [group2[0], group2[1], sum(group2[2:4], []), group2[4],
              group2[-1]]
    group3 = [['%s-%s' % (e, JVs[2][0]), '%s-%s' % (e, JVs[2][1])] for e in
              sum(Legs, [])] + [JVs[2]]
    group3 = [group3[0], group3[1], sum(group3[2:4], []), group3[4],
              group3[-1]]
    groups = group1 + group2 + group3 + Legs

    # assign colours to tag groups
    cdic = {groups[i][j]: plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            for i in range(len(groups)) for j in range(len(groups[i]))}
    df['c'] = df.index.to_list()
    df['c'] = df['c'].map(cdic)
    df.loc['default', 'c'] = 'w'

    return df, cdic


def align_yaxes(axes, min_values, max_values, signs):

    """
    Organises the y-axes corresponding to the plotted performance scores, so
    they will appear relative to one another, from the metric that displays the
    widest score range furthest to the left to the one that displays the
    smallest range closest to the data plot to the right.

    Arguments:
    ----------
    axes: matplotlib objects
        axes on which to apply the function

    min_values: list or array
        minimum of each statistical metric

    max_values: list or array
        maximum of each statistical metric

    signs: list
        positive if the lowest skill score indicates the highest performance,
        negative otherwise

    Returns:
    --------
    The rendered y-axes

    """

    # original attributes
    ticks = [ax.get_yticks() for ax in axes]
    bounds = [ax.get_ylim() for ax in axes]

    # align axes
    ticks_align = [ticks[i] - max_values[i] for i in range(len(axes))]

    # scale to 1 - 100
    ranges = [np.floor(-np.log10(e[-1] - e[0]) + 2.) for e in ticks]
    log_ticks = [ticks_align[i] * (10. ** ranges[i]) for i in range(len(axes))]

    # axes ticks into a single array, compute new ticks from it
    comb_ticks = np.concatenate(log_ticks)
    comb_ticks.sort()
    locator = MaxNLocator(50)  # a lot of bins to ensure nice aspect
    new_ticks = locator.tick_values(comb_ticks[0], comb_ticks[-1])
    new_ticks = [new_ticks / (10. ** ranges[i]) for i in range(len(axes))]
    new_ticks = [new_ticks[i] + max_values[i] for i in range(len(axes))]

    # find the new lower and upper bounds
    idx_l = 0
    idx_u = 0

    for i in range(len(new_ticks[0])):

        if any([new_ticks[j][i] > bounds[j][0] for j in range(len(axes))]):

            idx_l = i - 1

            break

    for i in range(len(new_ticks[0])):

        if all([new_ticks[j][i] > bounds[j][1] for j in range(len(axes))]):

            idx_u = i

            break

    # get new bounds given new ticks
    new_bounds = [(e[idx_l], e[idx_u]) for e in new_ticks]

    # set axes lims and spans at scale, ticks and ticklabels
    for i in range(len(axes)):

        axes[i].set_ylim(new_bounds[i])
        axes[i].spines['left'].set_bounds((min_values[i], max_values[i]))

        axes[i].set_yticks([min_values[i],
                            0.5 * (min_values[i] + max_values[i]),
                            max_values[i]])

        if signs[i] > 0:
            axes[i].set_yticklabels(['%.3f' % (min_values[i]),
                                     '%.3f' % (0.5 * (min_values[i] +
                                                      max_values[i])),
                                     '%.3f' % (max_values[i])])

        else:
            axes[i].set_yticklabels(['%.3f' % (1. - min_values[i]),
                                     '%.3f' % (1. - 0.5 * (min_values[i] +
                                                           max_values[i])),
                                     '%.3f' % (1. - max_values[i])])

    return


def skill_arrow(fig, ax, df, colours):

    """
    Draws an annotated coloured arrow that associates each model configuration
    with a different colour orders these from that with the lowest skill
    (lowest q-rank) to that with the highest skill (highest q-rank).

    Arguments:
    ----------
    fig: matplotlib objects
        fig on which to apply the function

    ax: matplotlib objects
        axis to annotate

    df: pandas dataframe
        dataframe containing information on the different model configurations'
        achieved quantile ranks

    colours:  dictionary
        colours associated with each model configuration / pairwise combination
        of Hleg and Nopt

    Returns:
    --------
    The annotated coloured arrow, below the main plot

    """

    # sort values by quantile rank
    df.set_index(df.columns.to_list()[0])
    df = df.sort_values(by=['rank'], ascending=False)

    # match ranks to colours in right order
    df['c'] = df.iloc[:, 0]
    df['c'] = df['c'].map(colours)
    df.loc[df[df.iloc[:, 0] == 'default'].index.to_list(), 'c'] = 'w'

    # add axis for the skill arrow
    ax2 = fig.add_axes([0.14225, -0.0625, 0.74, 0.03])

    # draw coloured squares
    bounds = np.arange(len(df) + 1)
    cb = mpl.colorbar.ColorbarBase(ax2,
                                   cmap=mpl.colors.ListedColormap(df['c']),
                                   norm=mpl.colors.BoundaryNorm(bounds,
                                                                len(df['c'])),
                                   ticks=bounds, spacing='proportional',
                                   extend='max', extendfrac='auto',
                                   orientation='horizontal')
    cb.set_ticks([])

    for i in range(5):  # annotate first five ranks

        ax2.text(bounds[-i - 1] - 0.5, 0.1, str(i + 1), va='bottom',
                 ha='center', c='w', size=12.)

    ax2.set_title('Quantile ranks', y=-1.275)

    # indicate where "skill" is good/bad
    ax.text(0.05, -0.15, 'Lower\nskill', va='center', ha='center',
            transform=ax.transAxes)
    ax.text(0.94, -0.15, 'Higher\nskill', va='center', ha='center',
            transform=ax.transAxes)

    return


def reading_key(fig, Legs, JVs):

    """
    Draws a 'matrix-like' legend that displays the colours associated with the
    different model configurations / pairwise combination of Hleg and Nopt.

    Arguments:
    ----------
    fig: matplotlib objects
        fig on which to apply the function

    Legs: nested list
        each sublist consists in an individual hydraulic legacies element to
        consider for the paiwise combinations of Hleg and Nopt

    JVs: nested list
        each sublist consists in an individual nitrogen optimisation element
        to consider for the paiwise combinations of Hleg and Nopt

    Returns:
    --------
    The reading key / legend as an inset in the main plot

    """

    # add axis for the reading key
    ax = fig.add_axes([0.7625, 0.65, 0.096, 0.128])

    # add coloured squares
    cmap = ['w'] + plt.rcParams['axes.prop_cycle'].by_key()['color'][::-1]
    ax.imshow(np.arange(len(cmap)).reshape((len(Legs) + 1, len(JVs) + 1),
              order='F'), cmap=mpl.colors.ListedColormap(cmap), aspect='auto')

    # add white grid to separate the squares
    ax.set_xticks(np.arange(len(JVs) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(Legs) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linewidth=2.)

    # add labels
    ax.set_xticks(np.arange(len(JVs) + 1))
    ax.set_xticklabels(['', r'\textit{35}' + '\n' + r'\textit{28}',
                        r'\textit{21}', r'\textit{14}' + '\n' + r'\textit{7}'],
                       va='center', ha='center', size=10)
    ax.set_yticks(np.arange(len(Legs) + 1))
    ax.set_yticklabels(['', r'\textit{180}', r'\textit{90$\vert$60}',
                        r'\textit{30}', r'\textit{14}'], va='center',
                       ha='center', size=10)

    # move and format labels
    ax.tick_params(which='both', bottom=False, left=False, labeltop=True,
                   labelbottom=False, pad=7.5)
    ax.tick_params(axis='y', pad=10)

    # axes names
    ax.set_xlabel(r'\textit{\mathrm{N$_{opt}$}}', rotation='horizontal',
                  va='center')
    ax.xaxis.set_label_coords(1.15, 1.15)
    ax.set_ylabel(r'\textit{\mathrm{H$_{leg}$}}', rotation='horizontal',
                  ha='center')
    ax.yaxis.set_label_coords(-0.2, -0.215)

    # reading key title
    ax.set_title('Configurations', x=-0.375, y=1.35, va='bottom', ha='left')

    # draw box around reading key
    box = mpatches.Rectangle((ax.axis()[0] - 1.75, ax.axis()[2] + 1.325),
                             (ax.axis()[1] - ax.axis()[0]) + 3.15,
                             (ax.axis()[3] - ax.axis()[2]) - 3.15,
                             fill=False, lw=1.25, ls='--', edgecolor='k')
    box = ax.add_patch(box)
    box.set_clip_on(False)

    return


def snake_plots(df1, df2, metrics, signs, Legs, JVs, cdic, fpath):

    """
    Plots each of the statistical metrics' values for all the model
    configurations / pairwise combination of Hleg and Nopt.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing information on the different model configurations'
        performance / skill as given by some chosen statistical metrics

    df2: pandas dataframe
        dataframe containing information on the different model configurations'
        achieved quantile ranks

    metrics: list
        statistical metrics to consider

    signs: list
        positive if the lowest skill score indicates the highest performance,
        negative otherwise

    Legs: nested list
        each sublist consists in an individual hydraulic legacies element to
        consider for the paiwise combinations of Hleg and Nopt

    JVs: nested list
        each sublist consists in an individual nitrogen optimisation element
        to consider for the paiwise combinations of Hleg and Nopt

    cdic:  dictionary
        colours associated with each model configuration / pairwise combination
        of Hleg and Nopt

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)performance.png' in the figure directory

    """

    # declare figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    axes = [ax1, ax2, ax3]

    # metrics specs
    markers = ['D', 'o', '>']
    symbols = [r'$\diamond$', r'$\circ$', r'$\triangleright$']

    for i in range(len(axes)):

        # sort in order of worst to best
        df1 = df1.sort_values(by=[metrics[i]], ascending=False)

        # variable to plot
        ivar = df1.columns.get_loc(metrics[i])

        # 'better than default' line
        idef = df1.index.get_loc('default')
        axes[i].plot(np.arange(len(df1.iloc[idef + 1:])) + idef + 1,
                     df1.iloc[idef + 1:, ivar], c='k')

        # position in pt size from data point
        if i == 0:
            x = 20.
            y = 17.5

        elif i == 1:
            x = 0.
            y = -22.5

        else:
            x = -35.
            y = 0.

        axes[i].annotate('default', xy=(idef, df1.iloc[idef, ivar]),
                         xytext=(x, y), xycoords='data',
                         textcoords='offset points', va='center', ha='center',
                         arrowprops=dict(arrowstyle='-', fc='w', ec='k'))

        # data points
        axes[i].scatter(np.arange(len(df1)), df1[metrics[i]],
                        marker=markers[i], c=df1['c'], zorder=10)

        # 5 best quantiles
        for j in range(5):

            # position in pt size from data point
            x = -signs[i] * signs[i - 1] * 15
            y = -signs[i] * signs[i - 1] * 15

            # change position so they are easy to read
            if ((i == 0) and (j in [1, 2, 4])) or ((i == 1) and
                                                   (j in [1, 2, 3])):
                x *= -1
                y *= -1

            # annotate with "wedged" pointing lines
            axes[i].annotate(str(j + 1), xy=(df1.index.get_loc(df2.iloc[j, 0]),
                             df1.loc[df2.iloc[j, 0], metrics[i]]),
                             xytext=(x, y), xycoords='data',
                             textcoords='offset points', va='center',
                             ha='center', c=df1.loc[df2.iloc[j, 0], 'c'],
                             arrowprops=dict(arrowstyle='wedge',
                                             color=df1.loc[df2.iloc[j, 0],
                                                           'c']))

        # add each individual axis' left spine
        axes[i].spines['left'].set_visible(True)
        axes[i].spines['left'].set_position(('axes', -i / 10.5))
        axes[i].yaxis.set_label_position('left')
        axes[i].yaxis.set_ticks_position('left')

    # change axes range and ticks to ensure appropriate scale
    align_yaxes(axes, [df1[e].min() for e in metrics],
                [df1[e].max() for e in metrics], signs)

    # position axes names
    for i in range(len(axes)):

        ymin, ymax = axes[i].get_ylim()  # range
        ypos = [(e - ymin) / (ymax - ymin) for e in axes[i].get_yticks()][0]

        axes[i].set_ylabel(r'%s (%s)' % (metrics[i].capitalize(), symbols[i]),
                           rotation='horizontal')
        axes[i].yaxis.set_label_coords(-i / 10., ypos - 0.065)

    # format x axis
    axes[i].spines['bottom'].set_visible(True)
    axes[i].spines['bottom'].set_bounds((0, 35))
    axes[0].set_xticks(np.arange(0, 36, 5))
    axes[0].set_xticklabels(np.arange(36, 0, -5), size=12)
    axes[0].set_xlabel(r'Absolute ranks', labelpad=10.)

    # add coloured skill arrow for the quantile ranks
    skill_arrow(fig, axes[0], df2, cdic)

    # add reading key for the different colours: which tag are they?
    reading_key(fig, Legs, JVs)

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
    default_plt_setup(spines=False)

    # specific setup
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size'] + 1.
    plt.rcParams['axes.labelsize'] = plt.rcParams['axes.titlesize']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['lines.linewidth'] = 1.
    plt.rcParams['scatter.edgecolors'] = 'k'  # marker style

    # user input
    main(dirpath(args.project))
