# -*- coding: utf-8 -*-

"""
Generate csv met data input, either random (from weather generator) or
from Fluxnet forcing, including parameters that the user can manually
change.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Forcing data csv generator"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (20.09.2019)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import warnings  # catch warnings
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# own modules
from TractLSM.SPAC import net_radiation  # radiation
from TractLSM.TraitOptimisation import optimal_kmax  # update kmax?

try:
    from general_utils import read_csv  # read in files
    from default_params import default_params  # forcing parameters
    from drivers_site_level import main as site_forcings
    import weather_generator as weather  # alternative forcings
    from calculate_solar_geometry import cos_zenith, top_atm_rad  # geom
    from calculate_solar_geometry import composite_LAI

except (ImportError, ModuleNotFoundError):
    from TractLSM.Utils.general_utils import read_csv
    from TractLSM.Utils.default_params import default_params
    from TractLSM.Utils.drivers_site_level import main as site_forcings
    import TractLSM.Utils.weather_generator as weather
    from TractLSM.Utils.calculate_solar_geometry import cos_zenith, top_atm_rad
    from TractLSM.Utils.calculate_solar_geometry import composite_LAI


# ======================================================================

def main(fname, d, Ndays=None, year=None):

    """
    Main function: either uses the weather generator to generate N days
                   of met data or writes a csv file containing
                   site-level FLUXNET forcings. Then, concatenates the
                   remaining forcing parameters which can be defined by
                   the user or be the default values.

    Arguments:
    ----------
    fname: string
        csv output filename (with path), preferably stored in the input/
        folder. The corresponding met (and flux and LAI) data must be
        stored in input/fluxsites/.

    d: class or pandas series
        contains the non-default parameters

    Ndays: int
        number of met data days to generate

    year: int
        selected year

    Returns:
    --------
    The comprehensive csv forcing file.

    """

    if Ndays is not None:
        ppfd, tdays, precip, vpd = get_met_data_wg(d, Ndays=Ndays)
        write_csv_wg(fname, d, ppfd, tdays, precip, vpd)

    elif year is not None:
        site_forcings(fname, year=year)

    else:
        site_forcings(fname)

    warnings.simplefilter('ignore',
                          category=pd.io.pytables.PerformanceWarning)
    add_vars_to_csv(fname, d)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def get_met_data_wg(d, Ndays):

    """
    Generates weather data for Ndays(s) data using the weather
    generator.

    Arguments:
    ----------
    d: class or pandas series
        contains the non-default parameters

    Ndays: int
        number of met data days to generate

    Returns:
    --------
    ppfd: array
        Nday diurnal course of the par [umol m-2 s-1]

    tdays: array
        Nday diurnal time course of the temperature [degC]

    precip: array
        Nday diurnal time course of rainfall [mm]

    vpd: array
        Nday diurnal course of the vapor pressure deficit [kPa]

    """

    # declare empty arrays for output
    ppfd = np.zeros((Ndays, 48))
    tdays = np.zeros((Ndays, 48))
    precip = np.zeros((Ndays, 48))
    vpd = np.zeros((Ndays, 48))

    for i in range(Ndays):

        doy = d.doy + float(i)

        if i == 0:
            tmin = d.tmin
            tmax = d.tmax
            rain_day = d.rain_day
            vpd15prev = d.vpd15prev
            vpd09 = d.vpd09
            vpd15 = d.vpd15
            vpd09next = d.vpd09next
            sw_rad_day = d.sw_rad_day

        # randomly change the inputs everyday
        tmin += np.random.uniform(-tmin / 5., tmin / 5.)
        tmax += np.random.uniform(-tmax / 5., tmax / 5.)

        # bound the inputs for them to still be physically acceptable
        if tmax >= d.tmax * 5. / 3.:
            tmin = d.tmin + np.random.uniform(-tmin / 5., tmin / 5.)
            tmax = d.tmax + np.random.uniform(-tmax / 5., tmax / 5.)

        if tmin <= - d.tmin * 15. / 2.:
            tmin = d.tmin + np.random.uniform(-tmin / 5., tmin / 5.)
            tmax = d.tmax + np.random.uniform(-tmax / 5., tmax / 5.)

        if tmax <= tmin:
            if tmin >= 18.:
                tmin = d.tmin + np.random.uniform(-tmin / 5., tmin / 5.)

            tmax = d.tmax + np.random.uniform(-tmax / 5., tmax / 5.)

        rain_day += np.random.uniform(-rain_day / 5., rain_day / 5.)

        # bound the inputs for them to still be physically acceptable
        if (rain_day < 0.) or (rain_day > d.rain_day * 20.):
            rain_day = d.rain_day

        if i != 0:
            vpd15prev = vpd15
            vpd09 = vpd09next

        vpd15 += np.random.uniform(-vpd15 / 5., vpd15 / 5.)

        vpd09next += np.random.uniform(-vpd09next / 5., vpd09next / 5.)

        # bound the inputs for them to still be physically acceptable
        if (vpd09 < 0.) or (vpd09 > 50.):
            vpd09 = d.vpd09

        if (vpd15 < 0.) or (vpd15 > 50.):
            vpd15 = d.vpd15

        if vpd09 >= vpd15:
            vpd09next = d.vpd09next + np.random.uniform(-vpd09next / 5.,
                                                        vpd09next / 5.)

        ppfd[i], tdays[i], precip[i], vpd[i] = weather.main(d.lat, d.lon, doy,
                                                            tmin, tmax,
                                                            rain_day,
                                                            vpd15prev,
                                                            vpd09, vpd15,
                                                            vpd09next,
                                                            sw_rad_day)

    return ppfd, tdays, precip, vpd


def write_csv_wg(fname, d, ppfd, tdays, precip, vpd):

    """
    Writes the forcing data generated by the weather generator into a
    csv.

    Arguments:
    ----------
    fname: string
        csv output filename (with path), preferably stored in the input/
        folder. The corresponding met (and flux and LAI) data must be
        stored in input/fluxsites/.

    d: class or pandas series
        contains the non-default parameters

    ppfd: array
        Nday diurnal course of the par [umol m-2 s-1]

    tdays: array
        Nday diurnal time course of the temperature [degC]

    precip: array
        Nday diurnal time course of rainfall [mm]

    vpd: array
        Nday diurnal course of the vapor pressure deficit [kPa]

    Returns:
    --------
    Creates the csv forcing file.

    """

    if d is None:
        d = default_params()

    elif type(d) == str:
        d = read_csv(d, drop_units=False)

    # generate the time series (WG.hours is N 0.5 h / day)
    WG = weather.WeatherGenerator(d.lat, d.lon)
    time = np.zeros(len(WG.hours) * ppfd.shape[0])
    step = WG.hours[1] - WG.hours[0]
    time[0] = d.doy * 24.

    for i in range(1, len(time)):

        time[i] = time[i-1] + step

    # column names
    varnames = ('doy', 'hod', 'coszen', 'PPFD', 'Tair', 'precip', 'VPD')
    units = ('[-]', '[h]', '[-]', '[umol m-2 s-1]', '[deg C]', '[mm d-1]',
             '[kPa]')
    ppfd = np.concatenate(ppfd, axis=0)  # same size as the time series
    tdays = np.concatenate(tdays, axis=0)
    precip = np.concatenate(precip, axis=0)
    vpd = np.concatenate(vpd, axis=0)

    # create the doy series that matches the time series
    doy = np.asarray(time / 24.).astype(int)
    hod = np.tile(np.arange(0.5, 24.5, 0.5), len(np.unique(doy)))

    cos_zen = [cos_zenith(doy[i], hod[i], d.lat, d.lon)
               for i in range(len(hod))]

    # is the sun up?
    ppfd[np.where(90. - np.degrees(np.arccos(cos_zen)) <= 0.)] = 0.
    ppfd[np.where(ppfd <= 50.)] = 0.  # min PAR (Ball & Berry, 1991)

    # write the csv
    these_headers = pd.MultiIndex.from_tuples(list(zip(varnames, units)))
    df = pd.DataFrame([doy, hod, cos_zen, ppfd, tdays, precip, vpd]).T
    df.columns = these_headers
    df.to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return


def update_site_params(df, p):

    """
    Updates a pandas series object used to store the model's parameters
    with the site-specific parameters.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all the data

    p: pandas series
        site parameters

    Returns:
    --------
    p: pandas series
        updated model's parameters (with site info)

    """

    ds = df.iloc[0]

    try:
        if str(ds['Vmax25']) != str(np.nan):
            p.Vmax25 = ds['Vmax25']

    except KeyError:
        pass

    try:
        if str(ds['albedo_l']) != str(np.nan):
            p.albedo_l = ds['albedo_l']

    except KeyError:
        pass

    try:
        if str(ds['max_leaf_width']) != str(np.nan):
            p.max_leaf_width = ds['max_leaf_width']

    except KeyError:
        pass

    try:
        if str(ds['P50']) != str(np.nan):
            p.P50 = ds['P50']

    except KeyError:
        pass

    try:
        if str(ds['P88']) != str(np.nan):
            p.P88 = ds['P88']

    except KeyError:
        pass

    try:
        if str(ds['ratiocrit']) != str(np.nan):
            p.ratiocrit = ds['ratiocrit']

    except KeyError:
        pass

    try:
        if str(ds['Psie']) != str(np.nan):
            p.Psie = ds['Psie']
            p.Ps = ds['Psie']

    except KeyError:
        pass

    return p


def composite_idealised_LAI(df):

    """
    Calculates the idealised LAI of the weighted sun/shade LAI, assuming
    no clouds, and site parameters (such as leaf emissivity).

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all the data

    Returns:
    --------
    LAI: float
        idealised composite LAI [m2 m-2]

    """

    # steps per day
    N = int(24. / (df['hod'].iloc[1] - df['hod'].iloc[0]))

    # growing season doys, hardcoded between April-Nov for 365 day year
    start = 91 * N
    end = 305 * N

    # extra-terrestrial incom. rad (Spitters et al., 1986), umol m-2 s-1
    Sxt = top_atm_rad(df['doy'].iloc[start:end], df['coszen'].iloc[start:end])

    # scatter the LAI, assuming PPFD is 0.75 of Sxt!
    LAI_comp = composite_LAI(df['doy'].iloc[start:end],
                             df['coszen'].iloc[start:end], Sxt, 0.75 * Sxt,
                             df['LAI'].iloc[start:end])

    # average new composite LAI per day
    LAI_comp = LAI_comp[:(len(LAI_comp) // N) * N].reshape(-1, N)
    LAI_comp = np.ma.masked_where(LAI_comp <= 0., LAI_comp)

    return np.mean(np.ma.mean(LAI_comp, axis=1))


def add_vars_to_csv(fname, d):

    """
    Appends the parameters that the user can manually change to the
    forcing csv file.

    Arguments:
    ----------
    fname: string
        csv output filename (with path), preferably stored in the input/
        folder. The corresponding met (and flux and LAI) data must be
        stored in input/fluxsites/.

    d: class or pandas series
        contains the non-default parameters

    Returns:
    --------
    The comprehensive csv forcing file.

    """

    mismatch = False
    df1 = read_csv(fname, drop_units=False)

    columns = ['Patm', 'u', 'CO2', 'O2', 'LAI', 'max_leaf_width', 'can_sat',
               'kcanint', 'Vmax25', 'Vmaxmin', 'Vmaxmax', 'gamstar25', 'Tref',
               'JV', 'JVmin', 'JVmax', 'Rlref', 'TRlref', 'Kc25', 'Ko25',
               'alpha', 'c1', 'c2', 'c3', 'c4', 'eps_l', 'eps_s', 'albedo_l',
               'albedo_ws', 'albedo_ds', 'tau_l', 'chi_l', 'kn', 'aj', 'bj',
               'kcat', 'ks', 'ccN', 'Ev', 'Ej', 'Egamstar', 'Ec', 'Eo',
               'deltaSv', 'deltaSj', 'Hdv', 'Hdj', 'P50', 'P88', 'kmax', 'r_k',
               'ratiocrit', 'g1', 'sfw', 'nPs', 'ground_area', 'Ztop',
               'Zbottom', 'root_beta', 'Psie', 'Ps', 'hyds', 'theta_sat', 'fc',
               'pwp', 'bch', 'r_soil']
    units = ['[kPa]', '[m s-1]', '[Pa]', '[kPa]', '[m2 m-2]', '[m]',
             '[mm d-1]', '[-]', '[umol s-1 m-2]', '[umol s-1 m-2]',
             '[umol s-1 m-2]', '[Pa]', '[deg C]', '[-]', '[-]', '[-]',
             '[umol s-1 m-2]', '[deg C]', '[Pa]', '[Pa]',
             '[mol(photon) mol(e-)-1]', '[-]', '[-]', '[-]', '[-]', '[-]',
             '[-]', '[-]', '[-]', '[-]', '[-]', '[-]', '[-]',
             '[umol molN-1 s-1]', '[umol molN-1 s-1]', '[molC molRub-1 s-1]',
             '[molN m2 s umolC-1]', '[-]', '[J mol-1]', '[J mol-1]',
             '[J mol-1]', '[J mol-1]', '[J mol-1]', '[J mol-1 K-1]',
             '[J mol-1 K-1]', '[J mol-1]', '[J mol-1]', '[-MPa]', '[-MPa]',
             '[mmol m-2 s-1 MPa-1]', '[-]', '[-]', '[kPa0.5]', '[-]', '[MPa]',
             '[m2]', '[m]', '[m]', '[1^(cm-1)]', '[MPa]', '[MPa]', '[m s-1]',
             '[m3 m-3]', '[m3 m-3]', '[m3 m-3]', '[-]', '[-]']

    try:
        df2 = pd.DataFrame([(d.Patm, d.u, d.CO2, d.O2, d.LAI, d.max_leaf_width,
                             d.can_sat, d.kcanint, d.Vmax25, d.Vmaxmin,
                             d.Vmaxmax, d.gamstar25, d.Tref, d.JV, d.JVmin,
                             d.JVmax, d.Rlref, d.TRlref, d.Kc25, d.Ko25,
                             d.alpha, d.c1, d.c2, d.c3, d.c4, d.eps_l, d.eps_s,
                             d.albedo_l, d.albedo_ws, d.albedo_ds, d.tau_l,
                             d.chi_l, d.kn, d.aj, d.bj, d.kcat, d.ks, d.ccN,
                             d.Ev, d.Ej, d.Egamstar, d.Ec, d.Eo, d.deltaSv,
                             d.deltaSj, d.Hdv, d.Hdj, d.P50, d.P88, d.kmax,
                             d.r_k, d.ratiocrit, d.g1, d.sfw, d.nPs,
                             d.ground_area, d.Ztop, d.Zbottom, d.root_beta,
                             d.Psie, d.Ps, d.hyds, d.theta_sat, d.fc, d.pwp,
                             d.bch, d.r_soil)], columns=columns)

        if str(d.kmax) == str(np.nan):  # add missing params
            d = d.append(pd.Series([25., 1., 1., d.albedo_ws, 1., 1.],
                         index=['Tair', 'VPD', 'scale2can', 'albedo_s',
                                'recovembo', 'legembo']))

            try:  # update site params using composite LAI
                if pd.notnull(np.sum(df2['LAI'])):
                    d.LAI = composite_idealised_LAI(df2)

            except KeyError:
                pass

            # add missing radiation params (no sensitivity to them)
            d = d.append(pd.Series([1000., net_radiation(d)],
                                   index=['PPFD', 'Rnet']))

            __, sol, __ = optimal_kmax(d, 'Farquhar', strategy='optimal')
            df2.kmax = sol[0]

    except AttributeError:  # always yield error if d is csv
        mismatch = True  # diffs if d is obj? all params must be present
        d2 = default_params()
        df2 = pd.DataFrame([(d2.Patm, d2.u, d2.CO2, d2.O2, d2.LAI,
                             d2.max_leaf_width, d2.can_sat, d2.kcanint,
                             d2.Vmax25, d2.Vmaxmin, d2.Vmaxmax, d2.gamstar25,
                             d2.Tref, d2.JV, d2.JVmin, d2.JVmax, d2.Rlref,
                             d2.TRlref, d2.Kc25, d2.Ko25, d2.alpha, d2.c1,
                             d2.c2, d2.c3, d2.c4, d2.eps_l, d2.eps_s,
                             d2.albedo_l, d2.albedo_ws, d2.albedo_ds, d2.tau_l,
                             d2.chi_l, d2.kn, d2.aj, d2.bj, d2.kcat, d2.ks,
                             d2.ccN, d2.Ev, d2.Ej, d2.Egamstar, d2.Ec, d2.Eo,
                             d2.deltaSv, d2.deltaSj, d2.Hdv, d2.Hdj, d2.P50,
                             d2.P88, d2.kmax, d2.r_k, d2.ratiocrit, d2.g1,
                             d2.sfw, d2.nPs, d2.ground_area, d2.Ztop,
                             d2.Zbottom, d2.root_beta, d2.Psie, d2.Ps, d2.hyds,
                             d2.theta_sat, d2.fc, d2.pwp, d2.bch, d2.r_soil)],
                           columns=columns)

    these_headers = list(zip(df2.columns, units))
    df2.columns = pd.MultiIndex.from_tuples(these_headers)

    # if data is already in df1, avoid over-writing it
    for ind in df2.columns.levels[0]:

        if ind in df1.columns.levels[0]:
            df2 = df2.drop([ind], axis=1)

    df = pd.concat([df1, df2], axis=1)
    original_columns = df.columns
    df.columns = df.columns.droplevel(level=1)

    # if params specified in site file or class object, overwrite df
    if mismatch:
        if '.csv' in d:
            df3 = read_csv(d, drop_units=False)

            try:  # site params in csv
                df3.columns = df3.columns.droplevel(level=1)
                df3 = df3.set_index('Site')
                sites = df3.index.tolist()
                sites = [str(e) for e in sites if str(e) != str(np.nan)]

                for site in sites:

                    if site in fname:
                        ds = df3.loc[site, :]

                    # rings or plots too
                    if 'Ring' in df3.columns:

                        for ring in df3['Ring'].unique():

                            if ring in fname:
                                ds = df3[df3['Ring'] == ring].loc[site, :]

                    elif 'Plot' in df3.columns:

                        for plot in df3['Plot'].unique():

                            if plot in fname:
                                ds = df3[df3['Ring'] == plot].loc[site, :]

            except KeyError:  # specific params in csv
                ds = df3.iloc[0]

            for ind in ds.index:

                if ((ind in df.columns) and not pd.isna(ds.loc[ind])):
                    df.iloc[0, df.columns.get_loc(ind)] = ds.loc[ind]

            # ini water potential cannot be above saturation: update
            if (df.iloc[0, df.columns.get_loc('Ps')] >
               df.iloc[0, df.columns.get_loc('Psie')]):
                df.iloc[0, df.columns.get_loc('Ps')] = \
                    df.iloc[0, df.columns.get_loc('Psie')]

            kmax = False  # kmax missing? replace with opt kmax

            try:
                if str(ds.loc['kmax']) == str(np.nan):
                    kmax = True

            except KeyError:
                kmax = True

            if kmax:
                d4 = default_params()

                # turn into pandas series
                attrs = vars(d4)
                d4 = {item[0]: item[1] for item in attrs.items()}
                d4 = pd.Series(d4)

                # site-level params
                d4 = update_site_params(df, d4)

                # add params missing from default params class
                d4 = d4.append(pd.Series([25., 1., 1., d4.albedo_ws, 1., 1.],
                               index=['Tair', 'VPD', 'scale2can', 'albedo_s',
                                      'recovembo', 'legembo']))

                try:  # update site params using composite LAI
                    if pd.notnull(np.sum(df['LAI'])):
                        d4.LAI = composite_idealised_LAI(df)

                except KeyError:
                    pass

                # add missing rad params (kmax insensitive to PAR)
                d4 = d4.append(pd.Series([1000.], index=['PPFD']))
                d4 = d4.append(pd.Series([net_radiation(d4)], index=['Rnet']))

                # get kmax
                __, sol, __ = optimal_kmax(d4, 'Farquhar', strategy='optimal')

                # update kmax,opt
                df.iloc[0, df.columns.get_loc('kmax')] = sol[0]

        else:  # class object

            for key in vars(d).keys():

                df.loc[:, key] = vars(d)[key]

    df.columns = pd.MultiIndex.from_tuples(original_columns)
    df.to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return
