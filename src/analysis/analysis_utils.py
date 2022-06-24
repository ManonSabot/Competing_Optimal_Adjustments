# -*- coding: utf-8 -*-

"""
Support functions to help analyse the results.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "useful ancillary functions"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (14.01.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for files, paths
import sys  # check for files, paths
import warnings  # ignore these warnings
import pandas as pd  # read/write dataframes, csv files

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # locate data
from TractLSM.Utils import read_csv  # read in data files

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================

def dirpath(project):

    """
    Finds the path to the output project repository specified.

    Arguments:
    ----------
    project: string
        project repository where the files to analyse are

    Returns:
    --------
    dir: string
        path for the project repository where the files to analyse are

    """

    basedir = os.path.dirname(get_main_dir())

    if project is not None:  # check that this project exists
        dir = os.path.join(os.path.join(os.path.join(basedir, 'output'),
                           'projects'), project)

        if not os.path.isdir(dir):
            raise NameError('The selected project does not exist.')

    else:
        dir = os.path.join(basedir, 'output')

    return dir


def concat_rings(fpath, endtag=None, keyword=None):

    """
    Concatenates dataframes of the different EucFACE "ring" plots
    together.

    Arguments:
    ----------
    fpath: string
        path to repository where the files to concatenate are

    endtag: string
        substring that should appear at the end of the file names

    keyword: string
        substring that should appear in the file names

    Returns:
    --------
    dfs: pandas dataframe
        dataframe containing all the individual "ring" plots' dataframes
        appended to one another

    """

    files = [e for e in os.listdir(fpath)]

    if endtag is not None:
        files = [e for e in files if e.endswith('%s.csv' % (endtag))]

    if keyword is not None:
        files = [e for e in files if keyword in e]

    for file in files:

        df, __ = read_csv(os.path.join(fpath, file))
        df['Ring'] = file.split('_')[0].split('EucFACE')[1]

        try:
            dfs = dfs.append(df, ignore_index=True)

        except UnboundLocalError:
            dfs = df.copy()

    # add dates in datetime format
    dfs['Date'] = pd.to_datetime(dfs['year'] * 1000. + dfs['doy'],
                                 format='%Y%j')

    return dfs
