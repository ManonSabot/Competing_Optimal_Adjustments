#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that populates the main model parameter file based on information
from text files in input/params/

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along this script.

"""

__title__ = "update model parameters"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (04.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"

# ======================================================================

# import general modules
import argparse  # read user input
import os  # check for files, paths
import sys  # make the TractLSM modules loadable
import pandas as pd  # read/write dataframes, csv files

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # locate data


# ======================================================================

def main(fname):

    """
    Main function: read in information from various text files and
                   accordingly fill up the main parameter file that is
                   used to parameterise the model input files

    Arguments:
    ----------
    fname: string
        parameter file used to parameterise the model input files (with
        path)

    Returns:
    --------
    The modified parameter file, typically 'site_params.csv' in input/

    """

    # path to files for where the estimated parameters are
    dir = os.path.join(os.path.dirname(get_main_dir()), 'input')
    idir = os.path.join(dir, 'params')

    # read in the param file
    df = pd.read_csv(fname).dropna(axis=0, how='all').squeeze()

    # populate the param file using various text files in input/params/
    for file in os.listdir(idir):

        if file.endswith('.txt'):
            ring = file.split('.txt')[0]

            df[df['Ring'] == ring] = match_param_value(df[df['Ring'] == ring],
                                                       os.path.join(idir,
                                                                    file))

    df.to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def contains_num(value):

    """
    Simple True/False check for a number within a string

    """

    return any([e.isdigit() for e in value])


def match_param_value(df, file):

    """
    Updates the parameters in df based on the data present in file.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe used to parameterise the model input files

    file: string
        text file that contains non-default parameter information on the
        parameters of a specific site / EucFACE ring (with path)

    Returns:
    --------
    df: pandas dataframe
        dataframe used to parameterise the model input files

    """

    f = open(file, 'r')
    lines = f.readlines()

    # info to keep
    k1 = ':'  # parameter names
    k2 = '+/-'  # calibrated parameters
    k3 = '(init'  # calibrated parameters
    k4 = '(fixed'  # calibrated parameters
    info = [e.split(k2)[0].split(k1) if (k2 in e and contains_num(e)) else
            e.split(k3)[0].split(k1) if (k3 in e and contains_num(e)) else
            e.split(k4)[0].split(k1) if (k4 in e and contains_num(e)) else
            e.split(k1) if (k1 in e and contains_num(e)) else ['']
            for e in lines]

    # remove end lines and formatting issues
    info = [e.strip('\n') for sub in info for e in sub if e != '']
    info = [e.replace(' ', '') if (':' in e) else e.strip() for e in info]
    params = [e for i, e in enumerate(info) if (i % 2 == 0)]
    values = [e for i, e in enumerate(info) if (i % 2 != 0)]

    # populate df
    for i in range(len(params)):

        if params[i] in df.columns:
            df[params[i]] = values[i]

    return df


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings
    description = "Populate the chosen parameter file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('paramfile', type=str, help='path to parameter file')
    args = parser.parse_args()

    main(args.paramfile)
