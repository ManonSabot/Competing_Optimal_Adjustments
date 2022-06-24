#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the differences between different model configurations
in terms of significant effect size on WUE, by contrast with the default
model.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "plot the effect of different configurations on model WUE"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (07.01.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths

import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
import itertools

# plotting modules
import matplotlib.pyplot as plt
import string  # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import amb_ele
from plot_utils import dryness

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from analysis.analysis_utils import dirpath  # locate data


# ======================================================================

def main(project, effect='diff'):

    """
    Main function: plot of the effect sizes of different model
                   configurations on WUE

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    effect: string
        the kind of effect size to plot: could be the average difference
        (diff), Welch's t (t), Glass’ delta (delta), etc.

    Returns:
    --------
    '(project_)WUE_effects.png' in the figure directory

    """

    # paths
    dir = os.path.dirname(os.path.dirname(project))
    eff = os.path.join(dir, '%s_effects.csv' % (os.path.basename(project)))

    # read in the file
    df = (pd.read_csv(eff).dropna(axis=0, how='all').dropna(axis=1, how='all')
          .squeeze())

    # keep the data for the ProfitMax model only
    df = df[df['model'] == 'pmax']

    # add dryness info by ring
    dryness(df, project.replace('output', 'input'))

    # prepare data for plotting and split into amb and ele rings
    amb, ele = effect_sizes(df, effect)

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project).replace('projects',
                                                           'figures'),
                          '%s_WUE_effects.png' % (os.path.basename(project)))

    # plot
    funnel_chart(amb, ele, figure)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def effect_sizes(df, effect):

    """
    Filters the effect size data to keep and plot, and orders the data
    depending on tag name and ring name, as well as separates the rings
    into aCO2 vs. eCO2 rings.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the effect sizes

    effect: string
        the kind of effect size to plot: could be the average difference
        (diff), Welch's t (t), Glass’ delta (delta), etc.

    Returns:
    --------
    amb: pandas dataframe
        effect sizes for the ambient CO2 rings, ordered by tag and ring

    ele: pandas dataframe
        effect sizes for the elevated CO2 rings, ordered by tag and ring

    """

    # statistically significant effects only
    (df.iloc[:, df.columns.get_loc('var') + 1: df.columns.get_loc('p')]
       .where(df['p'] < 0.05, 0., inplace=True))

    # restrict analysis to WUE
    df = df[df['var'].isin(['WUE'])]

    # comparison with the default
    df = df[np.logical_or(df['tag1'] == 'P-DSH-F-Rlow',
                          df['tag2'] == 'P-DSH-F-Rlow')]

    # the unique ref tag will be in only one of these two columns
    if len(df['tag1'].unique()) > 1:
        df.rename(columns={'tag1': 'tag'}, inplace=True)
        df['change'] = 100. * (df['avg1'] - df['avg2']) / df['avg2']

    else:
        df.rename(columns={'tag2': 'tag', 'tag1': 'tag2'}, inplace=True)
        df['change'] = 100. * (df['avg2'] - df['avg1']) / df['avg1']

    # more obvious tags + JV and Leg columns
    tags = [e.replace('P-DSH-F-Rlow-', '') for e in df['tag']]
    df['tag'] = np.asarray(tags)
    df['JV'] = [e.split('JV')[1] if ('JV' in e) else '0' for e in df['tag']]
    df['Leg'] = [e.split('Leg')[1].split('-JV')[0] if (('Leg' in e) and
                 ('-JV' in e)) else e.split('Leg')[1] if ('Leg' in e) else '0'
                 for e in df['tag']]

    # index tag order
    sc1 = [e for e in df['JV'].unique() if e != '0']
    sc2 = [e for e in df['Leg'].unique() if e != '0']
    sc1.sort(key=len)
    sc2.sort(key=len)
    scales = (['Leg%s-JV%s' % (e[0], e[1])
               for e in list(itertools.product(*[sc2, sc1]))] +
              ['JV%s' % (e) for e in sc1] + ['Leg%s' % (e) for e in sc2])
    scales.reverse()

    # index ring order
    rings = df.groupby(['ring'])['dryness'].mean().sort_values().index

    # group and re-order by tag and then ring
    df = df.groupby(['tag', 'ring']).mean()
    df = df.reindex(scales, level=0)
    df = df.reindex(rings, level=1)

    # select effect metric, keep relative change in means
    df = df[[effect, 'change']]

    # separate ambient from elevated rings
    amb = df[df.index.get_level_values(1).isin(amb_ele()[0])]
    ele = df[df.index.get_level_values(1).isin(amb_ele()[1])]

    return amb, ele


def additive(df):

    """
    Computes the additive effects of individual Hleg and JV tags that
    correspond to the various combinations of tags in the different
    model configurations tested.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing individual and combined effect sizes only

    Returns:
    --------
    df: pandas dataframe
        dataframe containing additive effect sizes only

    """

    single = [e for e in df.index.levels[0].to_list() if '-' not in e]
    comb = [e for e in df.index.levels[0].to_list() if '-' in e]

    for label in comb:  # replace the combined effects by added effects

        sub1 = df.loc[df.index.get_level_values(0) == label.split('-')[0]]
        sub1.index = sub1.index.droplevel(0)  # tag1
        sub2 = df.loc[df.index.get_level_values(0) == label.split('-')[1]]
        sub2.index = sub2.index.droplevel(0)  # tag2
        sub = sub1.add(sub2)  # additive effect

        for ring in df.index.get_level_values(1).unique():

            try:
                df.loc[(label, ring)] = sub.loc[ring]

            except IndexError:
                df.loc[(label, ring)] = np.nan

    for label in single:  # remove singular effects

        df.loc[label] = np.nan

    return df


def overlay_secondary_axis(ax, pos=-0.5, min=0., max=1., ticks=None,
                           where='left', visible=False, dash=False):

    """
    Creates a new axis which can be used to distribute text from top to
    bottom / left to write or to draw vertical / horizontal lines.

    Arguments:
    ----------
    ax: matplotlib object
        axis from which to create a secondary axis

    pos: float
        horizontal / vertical positioning of the new axis compared to ax

    min: float
        positioning of the bottom / left of the new axis compared to the
        bottom / left of ax (0 is the bottom / left of ax, 1 is its
        top / right)

    max: float
        positioning of the top / right of the new axis compared to the
        top / right of ax (0 is the bottom / left of ax, 1 is its top /
        right)

    ticks: list
        where to locate the y-axis / x-axis ticks

    where: string
        side of the axis that should be visible / on which the y-axis /
        x-axis ticks should appear

    visible: bool
        spines to render visible, on the side defined by "where"

    dash: bool
        whether the visible spine should be dashed or not

    Returns:
    --------
    nax: matplotlib object
        secondary axis

    """

    if (where == 'left') or (where == 'right'):
        nax = ax.twinx()
        nax.yaxis.set_ticks_position(where)

    else:  # bottom or top
        nax = ax.twiny()
        nax.xaxis.set_ticks_position(where)

    if visible:
        nax.spines[where].set_visible(True)

    # move the axis to its horizontal position and define its extent
    nax.spines[where].set_position(('axes', pos))
    nax.spines[where].set_bounds((min, max))

    if dash:  # dashes the axis line and makes it thinner
        nax.spines['left'].set_linestyle((0, (16, 16)))
        nax.spines['left'].set_linewidth(0.2)

    if ticks is not None:
        nax.set_yticks(ticks)

    return nax


def draw_brace(ax, span, text, where='left', sharpness=175., linewidth=1.,
               colour='k'):

    """
    Draws an annotated braces centered on an axis.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to draw braces

    span: list or array
        span of the brace relative to the axis

    text: string
        text to annotate besides the brace

    where: string
        side of the axis on which the brace should appear

    sharpness: float
        how pointy the middle of the brace should be

    linewidth: float
        width of the brace lines

    colour: string
        colour of the brace

    Returns:
    --------
    The drawn annotated brace

    """

    span = np.linspace(span[0], span[1], 501)
    halfspan = span[:500 // 2 + 1]
    halfbrace = (1. / (1. + np.exp(-sharpness * (halfspan - halfspan[0]))) +
                 1. / (1. + np.exp(-sharpness * (halfspan - halfspan[-1]))))
    brace = np.concatenate((halfbrace, halfbrace[-2:: -1]))

    if (where == 'left') or (where == 'bottom'):
        ax.plot(-brace, span, lw=linewidth, color=colour)
        ax.text(-4.5 * np.amin(brace), span[0] + (span[-1] - span[0]) / 2.,
                text, rotation=90, ha='center', va='center')

    else:
        ax.plot(brace, span, lw=linewidth, color=colour)
        ax.text(4.5 * np.amax(brace), span[0] + (span[-1] - span[0]) / 2.,
                text, ha='center', va='center')

    return


def funnel_chart(df1, df2, fpath):

    """
    Draws funnel bar charts of the WUE effect sizes ordered by model
    configuration and EucFACE's "ring" plots.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing the effect sizes for the ambient CO2 rings,
        ordered by tag and ring

    df2: pandas dataframe
        dataframe containing the effect sizes for the elevated CO2
        rings, ordered by tag and ring

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)WUE_effects.png' in the figure directory

    """

    # keep relative change aside, and drop from df
    changes = [df1['change'], df2['change']]
    df1.drop(columns=['change'], inplace=True)
    df2.drop(columns=['change'], inplace=True)

    # first get relative scales of dfs
    scs = [df1.sum() / (df1.sum() + df2.sum()),
           df2.sum() / (df1.sum() + df2.sum())]

    # declare the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 8),
                             gridspec_kw={'width_ratios': scs}, sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.125)

    for i, df in enumerate([df1, df2]):  # set up the data frames

        # relative sizes of combined effects
        df['rel'] = scs[i] * (df.abs() / df.abs().sum())
        df['rel2'] = additive(df['rel'].copy())

        # set positions to ensure the rings aren't on top of each other
        rings = df.index.get_level_values(1).unique().to_list()
        df['left'] = -0.5 * df['rel']  # center position

        # shift other rings
        idx1 = df.index.get_level_values(1) == rings[0]
        idx2 = df.index.get_level_values(1) == rings[1]
        idx3 = df.index.get_level_values(1) == rings[2]
        df.loc[idx1, 'left'] -= 0.5 * (df.loc[idx1, 'rel'].max() +
                                       df.loc[idx2, 'rel'].max())
        df.loc[idx3, 'left'] += 0.5 * (df.loc[idx2, 'rel'].max() +
                                       df.loc[idx3, 'rel'].max())

        if i > 0:  # adjust the scale to match both sides
            df.loc[idx1, 'left'] -= 0.00275 / scs[i - 1].iloc[0]
            df.loc[idx3, 'left'] += 0.0025 / scs[i - 1].iloc[0]

        else:
            df.loc[idx1, 'left'] -= 0.00275
            df.loc[idx3, 'left'] += 0.00275

        # where does the max change in the means occur?
        ichange = (changes[i].abs().groupby(df.index.get_level_values(1))
                             .idxmax().values)
        ichange = [list(e) for e in ichange]

        # loop over the labels
        labels = df.index.levels[0].to_list()
        cat = 0.
        yticks = []

        for j in range(len(labels)):

            label = labels[j]

            if j > 0:  # set the vertical position of effect size bar
                alabel = labels[j - 1]

                if '-' in label:
                    if label.split('-')[0] != alabel.split('-')[0]:
                        if np.isclose(cat, 0.2):  # jump to combined
                            cat += 1.

                        else:
                            cat += 0.2

                elif (label.translate(str.maketrans('', '', string.digits)) !=
                      alabel.translate(str.maketrans('', '', string.digits))):
                    cat += 0.2

            cat = round(cat, 1)  # avoid precision errors

            sub = df.loc[label]

            for __, row in sub.iterrows():

                pos = 0.5 * j + 0.5 + cat
                yticks += [pos]

                # plot the horizontal effect size bar
                axes[i].barh(pos, row['rel'] / (2. - 1.e-10), left=row['left'],
                             align='center', height=0.3, linewidth=0.)
                axes[i].barh(pos, row['rel'] / 2.,
                             left=row['left'] + row['rel'] / (2. - 1.e-10),
                             align='center', height=0.3, linewidth=0.)

                # mark places where sign is different
                if (np.sign(row.iloc[0]) !=
                   np.sign(df.mean(level=1).loc[row.name].iloc[0])):
                    axes[i].text(row['left'] + row['rel'] + 0.001, pos - 0.2,
                                 '*', ha='left', va='bottom')

                # mark the configuration yielding the greatest change
                if [str(label), str(row.name)] in ichange:
                    change = str(abs(round(changes[i].loc[(label, row.name)],
                                           1)))

                    if np.sign(row.iloc[0]) > 0.:
                        sign = '+'

                    else:
                        sign = '-'

                    if row.name in ['R1', 'R5']:
                        axes[i].text(row['left'] + row['rel'] + 0.0005,
                                     pos - 0.035, sign + change + r'$\%$',
                                     ha='left', va='center', c='w', size=10.5,
                                     bbox=dict(pad=0.3, fc='k'))

                    else:
                        axes[i].text(row['left'] + row['rel'] / 2.,
                                     pos - 0.035, sign + change + r'$\%$',
                                     ha='center', va='center', c='w',
                                     size=10.5, bbox=dict(pad=0.3, fc='k'))

                if '-' in label:
                    diff = (row['rel2'] - row['rel']) / 2.
                    axes[i].scatter([row['left'] - diff,
                                     row['left'] - diff + row['rel2']],
                                    [pos, pos], c='#343d46', zorder=10)

        # annotate each ring to indicate average sign of change
        for __, row in df.loc[labels[0]].iterrows():

            if np.isclose(df.sum(level=1).loc[row.name].iloc[0], 0.):
                sign = r'$\approx$'

            elif df.sum(level=1).loc[row.name].iloc[0] > 0.:
                sign = r'$\nearrow$'

            else:
                sign = r'$\searrow$'

            axes[i].text(row['left'] + 0.5 * row['rel'], -0.5,
                         '%s (%s)' % (row.name, sign), ha='center',
                         va='center')

        if i == 0:  # label plots
            title = r'aCO$_2$'

        else:
            title = r'eCO$_2$'

        axes[i].set_title(r'\textbf{(%s)} %s' %
                          (string.ascii_lowercase[i], title), loc='left')

    # separate the amb and ele rings by a light dashed line
    overlay_secondary_axis(axes[1], pos=-0.125, min=0.025, max=0.97725,
                           visible=True, dash=True, ticks=[])

    # separate the plots from the y-axis labels by a light dashed line
    overlay_secondary_axis(axes[0], pos=-0.05, min=0.025, max=0.97725,
                           visible=True, dash=True, ticks=[])

    # secondary y axes for Hleg & Nopt labels + offset them to the left
    skip = len([e for e in labels if (('JV' in e) and not ('-' in e))])
    yax1 = overlay_secondary_axis(axes[0], pos=-0.065, min=0.5 * (skip + 1.),
                                  max=pos + 0.2,
                                  ticks=np.unique(yticks)[skip:] + 0.01)
    yax1.set_ylim(axes[0].get_ylim())
    yax2 = overlay_secondary_axis(axes[0], pos=-0.13, min=0.3,
                                  max=pos - 0.5 * skip,
                                  ticks=np.unique(yticks) + 0.01)
    yax2.set_ylim(axes[0].get_ylim())

    # add Hleg & Nopt labels
    yax1.set_yticklabels([r'\textit{%s}' % (e.split('JV')[1]) if ('JV' in e)
                          else '' for e in labels[skip:]], ha='right')
    labels = labels[:skip] + [e if '21' in e else '' for e in labels[skip:]]
    yax2.set_yticklabels([r'\textit{%s}' % (e.split('Leg')[1].split('-JV')[0])
                          if (('Leg' in e) and ('-JV' in e))
                          else r'\textit{%s}' % (e.split('Leg')[1])
                          if ('Leg' in e) else '' for e in labels], ha='right')

    # indicate what they are both at the bottom and at the top
    yaxmin, yaxmax = axes[0].get_ylim()
    yax1.text(-0.018, -0.5, r'\textbf{\textit{\mathrm{N$_{opt}$}}}',
              va='center', ha='right')
    yax1.text(-0.018, yaxmax + 0.305, r'\textbf{\textit{\mathrm{N$_{opt}$}}}',
              va='center', ha='right')
    yax1.text(-0.0205, -0.5, r'\textbf{\textit{\mathrm{H$_{leg}$}}}',
              va='center', ha='right')
    yax1.text(-0.0205, yaxmax + 0.305, r'\textbf{\textit{\mathrm{H$_{leg}$}}}',
              va='center', ha='right')

    # add an axis for the annotated braces
    bax = fig.add_axes([-0.025, axes[0].get_position().y0, 0.03,
                        axes[0].get_position().y1 - axes[0].get_position().y0])
    bax.set_ylim(0., 1.)

    # plot annotated braces
    draw_brace(bax, [0.03, 0.26], 'Singular effects')
    draw_brace(bax, [0.34, 0.965], 'Joint effects')

    # remove unwanted ticks
    axes[0].axis('off')
    axes[1].axis('off')
    bax.axis('off')

    # save figure
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
    parser.add_argument('-e', '--effect', type=str, default='diff',
                        help='effect size metric / test to use')
    args = parser.parse_args()

    # default setup
    default_plt_setup(colours=['#748891'], spines=False)

    # specific setup
    plt.rcParams['axes.titlepad'] = 0.3  # aCO2/eCO2 near plots
    plt.rcParams['scatter.marker'] = '|'  # marker style
    plt.rcParams['ytick.major.pad'] = 10  # ticks to ticklabels' pad
    plt.rcParams['ytick.major.size'] = 0.  # hide ticks

    # user input
    main(dirpath(args.project), args.effect)
