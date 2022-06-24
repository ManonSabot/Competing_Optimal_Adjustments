#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that checks different model configurations/output files for
significant differences.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "differences between model configurations"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (08.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths
import warnings  # ignore these warnings
import numpy as np  # array manipulations, math operators
import itertools
import pandas as pd  # read/write dataframes, csv files
import researchpy  # compute effect sizes

# own modules
from analysis_utils import dirpath

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import read_csv  # read in data files

# ignore these warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(project):

    """
    Main function: Generates '(project_)effects.csv' in the output
                   directory, which contains information on the
                   differences between different configurations of the
                   model.

    Arguments:
    ----------
    project: string
        project repository where the files to analyse are (with path)

    Returns:
    --------
    '(project_)effects.csv' in the output directory

    """

    # combinations of files to test against one another
    combis = find_comparison_combis(project)

    # go over the combinations
    for combi in combis:

        differences(project, combi)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def find_comparison_combis(project):

    """
    Finds all combinations of the default model configuration and all
    other model configurations tested.

    Arguments:
    ----------
    project: string
        project repository where the files to analyse are (with path)

    Returns:
    --------
    combis: nested lists or arrays
        files to compare, nested two by two, in the combinations order

    """

    files = [e for e in os.listdir(project) if
             ('DSH' in e and e.endswith('.csv'))]
    uniques = np.unique([e.split('_')[0] for e in files])

    combis = []

    for unique in uniques:

        sub = sorted([e for e in files if unique in e])
        sub.sort(key=len)
        sub_combis = list(itertools.product([sub[0]], sub[1:]))
        combis += list(list(sorted(e)) for e in sub_combis)

    return combis


def differences(project, combination):

    """
    Computes statistical differences between variables of different
    configurations of the model.

    Arguments:
    ----------
    project: string
        project repository where the files to analyse are (with path)

    combination: list or array
        combination of two files to compare

    Returns:
    --------
    Generates '(project_)effects.csv' in the output directory

    """

    # is the difference file already present?
    fdiff = os.path.join(os.path.dirname(os.path.dirname(project)),
                         '%s_effects.csv' % os.path.basename(project))

    if os.path.isfile(fdiff):  # append combination to existing file
        fd = (pd.read_csv(fdiff).dropna(axis=0, how='all')
                .dropna(axis=1, how='all').squeeze())
        fd.index = fd['combi']

    else:  # create difference file
        fd = (pd.DataFrame([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan]).T)
        fd.columns = ['ring', 'tag1', 'tag2', 'model', 'var', 'avg1', 'avg2',
                      'diff', 't', 'd', 'g', 'delta', 'p', 'fname1', 'fname2',
                      'combi']
        fd.index = fd['combi']

    # general info about the combination files
    ring = combination[0].split('_')[0].split('EucFACE')[1]
    tag1 = combination[0].split('_')[1].split('.csv')[0]
    tag2 = combination[1].split('_')[1].split('.csv')[0]

    # read in the files
    df, __ = read_csv(os.path.join(project.replace('output', 'input'),
                                   'EucFACE%s_model_drivers.csv' % (ring)))
    df1, __ = read_csv(os.path.join(project, combination[0]))
    df2, __ = read_csv(os.path.join(project, combination[1]))

    # add WUE to the variables
    columns = df1.columns.to_list()
    models = np.unique([e.split('(')[1].split(')')[0] for e in columns
                        if '(' in e])

    for mod in models:

        df1['WUE(%s)' % (mod)] = df1['An(%s)' % (mod)] / df1['E(%s)' % (mod)]
        df2['WUE(%s)' % (mod)] = df2['An(%s)' % (mod)] / df2['E(%s)' % (mod)]

    # sort out models and variables to consider
    columns = df1.columns.to_list()
    columns = np.unique([e.split('(')[0] for e in columns if
                         (any(m in e for m in models) and ('Rublim' not in e)
                          and ('Eci' not in e) and ('Nc' not in e) and
                          ('Ne' not in e) and ('Nr' not in e) and
                          ('Ns' not in e) and ('VCsun' not in e) and
                          ('VCsha' not in e))])

    for mod in models:

        # remove times when no photo happens because it's nighttime
        df1b = df1[df['PPFD'] > 50.]
        df2b = df2[df['PPFD'] > 50.]

        for col in columns:

            try:  # compute Welch's t, Hedges’ g, Glass’ delta
                __, res = researchpy.ttest(df1b['%s(%s)' % (col, mod)],
                                           df2b['%s(%s)' % (col, mod)],
                                           equal_variances=False)

                # append to df
                fd = fd.append(pd.Series({'ring': ring,
                                          'tag1': tag1,
                                          'tag2': tag2,
                                          'model': mod,
                                          'var': col,
                                          'avg1': (df1b['%s(%s)' % (col, mod)]
                                                   .mean()),
                                          'avg2': (df2b['%s(%s)' % (col, mod)]
                                                   .mean()),
                                          'diff': res.iloc[0]['results'],
                                          't': res.iloc[2]['results'],
                                          'd': res.iloc[6]['results'],
                                          'g': res.iloc[7]['results'],
                                          'delta': res.iloc[8]['results'],
                                          'p': res.iloc[3]['results'],
                                          'fname1': combination[0],
                                          'fname2': combination[1],
                                          'combi': '%s_%s' % (tag1, tag2)},
                                         name='%s_%s' % (tag1, tag2)))

            except KeyError:
                pass

    # save output
    fd.dropna(inplace=True)
    fd.drop_duplicates(inplace=True)
    fd.to_csv(fdiff, index=False, na_rep='', encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-R', '--project', type=str,
                        help='folder containing the files to analyse')
    args = parser.parse_args()

    # execute main
    main(dirpath(args.project))
