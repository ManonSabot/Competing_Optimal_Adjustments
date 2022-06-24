#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that converts data stored in the .npy format to data in the .csv
format.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along this script.

"""

__title__ = "data storage conversion"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (12.05.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # locate data


# ======================================================================

def main():

    """
    Main function: convert a series of .npy files to the more human
                   readable .csv format

    """

    # where .npy data are stored
    for folder in ['obs', 'canopy']:

        # path to files
        fdir = os.path.join(os.path.join(os.path.dirname(get_main_dir()),
                                         'input'), folder)
        files = [e for e in os.listdir(fdir) if e.endswith('.npy')]

        for file in files:  # convert to .csv

            df = pd.DataFrame(np.load(os.path.join(fdir, file),
                              allow_pickle=True))
            df.drop(columns=['index'], inplace=True)
            df.to_csv(os.path.join(fdir, file.replace('.npy', '.csv')),
                      index=False, na_rep='', encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    main()
