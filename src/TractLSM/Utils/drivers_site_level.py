# -*- coding: utf-8 -*-

"""
Take site-level eddy-covariance met data and generate a forcing file.
In the cases where flux data is also available, this is added to the
forcing file (but will not force the model), for validation purposes
after the model has finished running. If LAI info is also present, this
is added to the forcing file and is used to run the model.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* Asner, G. P., Scurlock, M. O., & Hicke, J. A. (2003). Global Ecology &
  Biogeography, 12, 191–205.
* Jackson, R. B., Canadell, J., Ehleringer, J. R., Mooney, H. A., Sala,
  O. E., & Schulze, E. D. (1996). A global analysis of root distributions
  for terrestrial biomes. Oecologia, 108(3), 389-411.
* Monteith, J. L., & Unsworth, M. H. (1990). Principles of environmental
  physics. Arnold. SE, London, UK.

"""

__title__ = "Site-level forcings (met, flux, and LAI info in csv)"
__author__ = ["Manon E. B. Sabot", "Martin G. De Kauwe"]
__version__ = "3.0 (22.05.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import os  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# own modules
from TractLSM import conv, cst  # unit converter & general constants

try:
    from general_utils import read_csv, read_netcdf  # read in files
    from calculate_solar_geometry import cos_zenith  # solar geometry

except (ImportError, ModuleNotFoundError):
    from TractLSM.Utils.general_utils import read_csv, read_netcdf
    from TractLSM.Utils.calculate_solar_geometry import cos_zenith


# ======================================================================

def main(fname, year=None):

    """
    Main function: reads fluxnet data in netcdf format (open_file),
                   retrieves the forcing of all years or a specific
                   year (generate_forcing). If the fluxes are
                   present, Qle, NEE, and GPP are added as
                   information to the forcing (however, they will
                   not be used to run the model).

    Arguments:
    ----------
    fname: string
        csv forcing filename (with path), preferably stored in the
        input/ folder. The corresponding met (and flux and LAI) data
        must be stored in input/fluxsites/.

    year: int
        selected year

    Returns:
    --------
    A csv file with the name site_met_and_plant_data_actual.csv or
    site_met_and_plant_data_spinup.csv in the input/(projects/project/)
    folder.

    """

    # separate the folder name from the csv file name
    fname_right = ''.join(fname.split(os.path.sep)[-1:])
    site = fname_right.split('_')[:-1][0]

    fname_left = str(os.path.sep).join(fname.split(os.path.sep)[:-1])
    fname_left2 = fname_left

    # if Rings, e.g. at EucFACE, deal with this
    site2 = site
    rings = None

    if 'EucFACE' in site:
        amb_rings = ['R2', 'R3', 'R6']
        ele_rings = ['R1', 'R4', 'R5']

        if site[-1].isdigit() and (site[-2] == 'R'):
            rings = ['R%s' % (site.split('R')[1])]

            if site[-2:] in amb_rings:
                site2 = '%samb' % (site.split('R')[0])

            elif site[-2:] in ele_rings:
                site2 = '%sele' % (site.split('R')[0])

        elif 'amb' in site:
            rings = amb_rings

        elif 'ele' in site:
            rings = ele_rings

    # in case this is a project, remove project name
    if ((not fname_left2.endswith('input%s' % str(os.path.sep))) and
       (not fname_left2.endswith('input'))):
        if fname_left2.endswith(str(os.path.sep)):
            fname_left2 = (str(os.path.sep)
                           .join(fname_left2.split(str(os.path.sep))[:-1]))

        fname_left2 = (str(os.path.sep).join((str(os.path.sep)
                       .join(fname_left2.split(str(os.path.sep))[:-1])
                       .split(str(os.path.sep))[:-1])))

    try:  # read in the forcing file
        fname = os.path.join(os.path.join(fname_left, 'atm'),
                             '%s.nc' % (site2))
        df = open_file(fname)

    except FileNotFoundError:  # deal with project
        fname = os.path.join(os.path.join(os.path.dirname(
                                          os.path.dirname(fname_left)), 'atm'),
                             '%s.nc' % (site2))
        df = open_file(fname)

    # generate forcing file with the 'right' info.
    ofname = os.path.join(fname_left, '%s_model_drivers.csv' % (site))
    generate_forcing(ofname, df)
    df = read_csv(ofname, drop_units=False)

    try:  # add the corresponding LAI
        fLAI = os.path.join(os.path.join(fname_left, 'canopy'),
                            'smooth_PAI_drift_corrected.csv')
        df1 = (pd.read_csv(fLAI).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    except FileNotFoundError:  # deal with project
        fLAI = os.path.join(os.path.join(os.path.dirname(os.path.dirname(
                                         fname_left)), 'canopy'),
                            'smooth_PAI_drift_corrected.csv')
        df1 = (pd.read_csv(fLAI).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    if 'EucFACE' in site:  # from PAI to LAI
        df1['LAI'] -= 0.8

    df1 = LAI(df1, rings=rings)

    # for every year and doy in df1, append the corresponding LAI
    df = df.merge(df1, how='left', on=[('year', '[-]'), ('doy', '[-]')])

    if rings is None:  # add soil moisture profile
        try:
            fsm = os.path.join(os.path.join(fname_left, 'soil'),
                               'EucFACE_sw.csv')
            df1 = read_csv(fsm, drop_units=False)

        except FileNotFoundError:
            fsm = os.path.join(os.path.join(os.path.dirname(os.path.dirname(
                               fname_left)), 'soil'), 'EucFACE_sw.csv')
            df1 = read_csv(fsm, drop_units=False)

    else:
        try:
            fsm = os.path.join(os.path.join(fname_left, 'soil'),
                               '%s_sw.csv' % (site))
            df1 = read_csv(fsm, drop_units=False)

        except FileNotFoundError:
            fsm = os.path.join(os.path.join(os.path.dirname(os.path.dirname(
                               fname_left)), 'soil'), '%s_sw.csv' % (site))
            df1 = read_csv(fsm, drop_units=False)

    # for every year and doy in df1, append the soil moisture
    df = df.merge(df1, how='left', on=[('year', '[-]'), ('doy', '[-]')])
    df.drop_duplicates(inplace=True)
    df[['sw', 'sw0', 'Ps']] = df[['sw', 'sw0', 'Ps']].bfill()

    # save the forcing file
    df.to_csv(ofname, index=False, na_rep='', encoding='utf-8')

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def qair_to_vpd(qair, tair, press):

    """
    Calculates the saturation vapour pressure at a specific temperature
    tair as given in Monteith & Unsworth, 1990. Then calculates the
    actual air vapour pressure, and finally, the VPD.

    Arguments:
    ----------
    qair: array or float
        near surface specific humidity [kg kg-1]

    tair: array or float
        near surface air temperature [degK]

    press: array or float
        surface air pressure [kPa]

    Returns:
    --------
    The vapour pressure deficit [kPa] at T.

    """

    # saturation vapor pressure (Tetens eq.)
    T = tair - conv.C_2_K  # degC
    es = 0.61078 * np.exp(17.27 * T / (T + 237.3))  # kPa

    # actual vapour pressure
    RH = qair * cst.Rv * (press - es) / (cst.Rd * es)  # not %, 0.-1.
    ea = es * RH  # kPa

    return es - ea


def open_file(fname):

    """
    Performs unit conversions on the data, and a variable conversion
    from near surface specific humidity to vapour pressure deficit if
    met data, saves the flux data as it is.

    Arguments:
    ----------
    fname: string
        csv output filename (with path), preferably stored in the input/
        folder. The corresponding fluxnet data must be stored in
        input/fluxsites/

    Returns:
    --------
    df: pandas dataframe
        df containing the met forcing data or flux data

    """

    df = read_netcdf(fname)  # from netcdf to df

    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.droplevel()  # drop soil layers

    # add solar zenith angle for two-leaf model
    hod = (np.array(df.index.hour).astype(np.float) +
           np.array(df.index.minute).astype(np.float) / 60.)
    df['coszen'] = cos_zenith(np.array(df.index.dayofyear)
                                .astype(np.float), hod,
                              df.iloc[(0, df.columns.get_loc('latitude'))],
                              df.iloc[(0, df.columns.get_loc('longitude'))])

    df['PPFD'] = df['SWdown'] * conv.SW_2_PAR  # umol m-2 s-1
    df['PPFD'].where(90. - np.degrees(np.arccos(df['coszen'])) > 0., 0.,
                     inplace=True)  # the sun isn't up

    try:
        df['Rainf'] *= conv.SEC_2_DAY  # mm d-1

    except KeyError:
        df = df.rename(columns={'Precip': 'Rainf'})
        df['Rainf'] *= conv.SEC_2_DAY  # mm d-1

    try:
        df['PSurf'] *= conv.FROM_MILI  # Pa to kPa

    except KeyError:  # use barometric formula
        df['PSurf'] = (101.325 * (df['Tair'] / (df['Tair'] +
                       cst.Lb * df['elevation'])) ** (cst.g0 * cst.Mair *
                       conv.FROM_MILI / (cst.R * cst.Lb)))  # kPa

    try:
        df['VPD'] *= 0.1  # from hPa to kPa

    except KeyError:
        df['VPD'] = qair_to_vpd(df['Qair'], df['Tair'], df['PSurf'])

    df['VPD'].where(df['VPD'] > 0.05, inplace=True)  # NaNs
    df['VPD'].interpolate(inplace=True)  # fill NaNs

    df['Tair'] -= conv.C_2_K  # degC
    df['CO2air'] *= conv.FROM_MILI * df['PSurf']  # ppm to Pa

    return df


def generate_forcing(ofname, df):

    """
    Saves the unaltered data (actual year) in a csv file with the
    appropriate structure.

    Arguments:
    ----------
    ofname: string
        name of the csv which must be saved containing path (default is
        input/)

    df: pandas dataframe
        met and flux data for the selected year

    Returns:
    --------
    Saves a csv file with a name of the form
    site_met_and_plant_data_actual.csv in the input/ folder.

    """

    # replace column names and co.
    df.rename(columns={'Rainf': 'precip', 'PSurf': 'Patm', 'Wind': 'u',
                       'CO2air': 'CO2'}, inplace=True)

    # variables and corresponding units for the two headers
    ovars = ['year', 'doy', 'hod', 'coszen', 'PPFD', 'Tair', 'precip', 'VPD',
             'Patm', 'u', 'CO2']
    ounits = ['[-]', '[-]', '[h]', '[-]', '[umol m-2 s-1]', '[deg C]',
              '[mm d-1]', '[kPa]', '[kPa]', '[m s-1]', '[Pa]']

    # year, hod and doy
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear
    df['hod'] = df.index.hour + df.index.minute / 60. + 0.5

    # restrict to relevant vars, add multi-header and save
    df = df[ovars]
    df.columns = pd.MultiIndex.from_tuples(list(zip(ovars, ounits)))
    df.to_csv(ofname, index=False, na_rep='', encoding='utf-8')

    return


def LAI(df, rings=None):

    # variables and corresponding units for the two headers
    ovars = ['year', 'doy', 'LAI']
    ounits = ['[-]', '[-]', '[m2 m-2]']

    if rings is not None:  # select rings to consider
        df = df[df['Ring'].isin(rings)]

    # date as index
    df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'], dayfirst=True)
    df = df.groupby('Date').mean()

    # year and doy
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear

    # restrict to relevant vars, add multi-header and save
    df = df[ovars]
    df.columns = pd.MultiIndex.from_tuples(list(zip(ovars, ounits)))

    return df
