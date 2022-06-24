# -*- coding: utf-8 -*-

"""
Default parameter class, necessary to run the model.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* ten Berge, Hein FM. Heat and water transfer in bare topsoil and lower
  atmosphere. No. 33. Pudoc, 1990.
* Campbell, G. S., & Norman, J. M. “An Introduction to Environmental
  Biophysics” 2nd Edition, Springer-Verlag, New York, 1998.
* Choat, B., Jansen, S., Brodribb, T. J., Cochard, H., Delzon, S.,
  Bhaskar, R., ... & Jacobsen, A. L. (2012). Global convergence in the
  vulnerability of forests to drought. Nature, 491(7426), 752.
* Evans, J. R. (1989). Photosynthesis-the dependence on nitrogen
  partitioning. Causes and consequences of variation in growth rate and
  productivity of higher plants, 159-174.
* Evans, J. R. (1993). Photosynthetic acclimation and nitrogen
  partitioning within a lucerne canopy. I. Canopy characteristics.
  Functional Plant Biology, 20(1), 55-67.
* Gale, M. R., & Grigal, D. F. (1987). Vertical root distributions of
  northern tree species in relation to successional status. Canadian
  Journal of Forest Research, 17(8), 829-834.
* Kattge, J., & Knorr, W. (2007). Temperature acclimation in a
  biochemical model of photosynthesis: a reanalysis of data from 36
  species. Plant, cell & environment, 30(9), 1176-1190.
* Medlyn, B. E. (1996). The optimal allocation of nitrogen within the C3
  photosynthetic system at elevated CO2. Functional Plant Biology,
  23(5), 593-603.
* Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley,
  P. C., Kirschbaum, M. U. F., ... & Wang, K. (2002). Temperature
  response of parameters of a biochemically based model of
  photosynthesis. II. A review of experimental data. Plant, Cell &
  Environment, 25(9), 1167-1179.
* Peltoniemi, M. S., Duursma, R. A., & Medlyn, B. E. (2012). Co-optimal
  distribution of leaf nitrogen and hydraulic conductance in plant
  canopies. Tree Physiology, 32(5), 510-519.

"""

__title__ = "default parameter class necessary to run the model"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (30.07.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

class default_params(object):  # default inputs needed to run model

    def __init__(self):

        # information used by the weather generator
        self.doy = 180.  # day of the year
        self.tmin = 2.  # degC
        self.tmax = 24.
        self.rain_day = 2.  # mm d-1
        self.vpd15prev = 3.4
        self.vpd09 = 1.4
        self.vpd15 = 2.3
        self.vpd09next = 1.8
        self.sw_rad_day = 1080. * 10.  # W m-2, for 10 daylight hours
        self.Patm = 101.325  # kPa
        self.u = 2.  # m s-1

        # location matters for the zenithal angle
        self.lat = 38.569120
        self.lon = -80.018519

        # gas concentrations
        self.CO2 = 37.  # Pa, by default 375 ppm as in ~2005
        self.O2 = 20.73  # kPa

        # canopy structure & interception
        self.LAI = 2.  # m2 m-2
        self.max_leaf_width = 0.05  # m
        self.can_sat = 10.  # min. rainfall for interception (mm d-1)
        self.kcanint = 0.7  # interception attenuation coefficient (-)

        # photosynthesic parameters
        self.Vmax25 = 100.  # max carboxyl rate @ 25 degC (umol m-2 s-1)
        self.Vmaxmin = 10.  # useful if the leaf N coord is activated
        self.Vmaxmax = 180.  # useful if the leaf N coord is activated
        self.gamstar25 = 4.22  # CO2 compensation point @ 25 degC (Pa)
        self.Tref = 25.  # ref T for Vmax25, gamstar, deltaS, Hd
        self.JV = 1.67  # Jmax25 to Vmax25 ratio
        self.JVmin = 1.  # useful if the leaf N coord is activated
        self.JVmax = 3.  # useful if the leaf N coord is activated
        self.Rlref = self.Vmax25 * 0.015  # resp @ TRlref (umol m-2 s-1)
        self.TRlref = 25.  # T for the ref keaf respiration (degC)
        self.Kc25 = 39.96  # Michaelis-Menten cst for carboxylation (Pa)
        self.Ko25 = 27.48  # Michaelis-Menten cst for oxygenation (kPa)
        self.alpha = 0.3  # quantum yield photo (mol(photon) mol(e-)-1)

        # Farquhar model
        self.c1 = 0.7  # curvature of light response
        self.c2 = 0.99  # transition Je vs Jc (Peltoniemi et al.)

        # Collatz model
        self.c3 = 0.83  # curvature of light response
        self.c4 = 0.93  # transition Je vs Jc

        # energies of activation
        self.Ev = 60000.  # Vcmax, J mol-1
        self.Ej = 30000.  # Jmax, J mol-1
        self.Egamstar = 37830.  # gamstar, J mol-1
        self.Ec = 79430.  # carboxylation, J mol-1
        self.Eo = 36380.  # oxygenation, J mol-1

        # inhibition at higher temperatures (Kattge & Knorr)
        self.deltaSv = 650.  # Vmax entropy factor (J mol-1 K-1)
        self.deltaSj = 650.  # Jmax entropy factor (J mol-1 K-1)
        self.Hdv = 200000.  # Vmax decrease rate above opt T (J mol-1)
        self.Hdj = 200000.  # Jmax decrease rate above opt T (J mol-1)

        # relating to light / rad (C & N is Campbell & Norman)
        self.eps_l = 0.97  # leaf emiss. LW (Table 11.3 C & N)
        self.eps_s = 0.945  # soil emiss. LW (Table 11.3 C & N)
        self.albedo_l = 0.062  # leaf SW vis (CABLE)
        self.albedo_ws = 0.1  # wet soil SW vis (Table 11.2 C & N)
        self.albedo_ds = 0.25  # dry soil SW vis (ten Berge)
        self.tau_l = 0.05  # leaf transmis. SW vis (CABLE)
        self.chi_l = 9.99999978e-3  # leaf angle dist (spherical = 0)
        self.kn = 0.001  # extinction coef. of nitrogren (CABLE)

        # leaf nitrogen (Evans; Medlyn)
        self.aj = 15870.  # e-transport to N distri (umol (molN)-1 s-1)
        self.bj = 2775.  # e-transport to N distri (umol (molN)-1 s-1)
        self.kcat = 24.  # rub. activ. @ 25 degC (molC (molRub)-1 s-1)
        self.ks = 1.25e-4  # solub. protein activ. (molN m2 s umol-1)
        self.ccN = 0.076 / 25.  # coef. for chlorophyll N (unitless)

        # hydraulics
        self.P50 = 6.6  # xylem pressure at P50 (-MPa) - J. virginiana
        self.P88 = 10.5  # same at P88 (-MPa) (Choat et al.)
        self.kmax = 1.  # max plant hydr cond / LAI (mmol m-2 s-1 MPa-1)
        self.r_k = 1.  # recovery of embolism (unitless)
        self.ratiocrit = 0.05  # Pcrit; stom control? kcrit = N%(kmax)

        # stomatal conductance
        self.g1 = 2.35  # sensitivity of stomatal conduc to An (kPa0.5)
        self.sfw = 0.2  # sensitivity power factor on fw (unitless)
        self.nPs = 0.5  # offset on the soil water potential (MPa)

        # soil
        self.ground_area = 1.  # m2
        self.Ztop = 0.02  # top soil layer depth (m)
        self.Zbottom = 1.  # bottom soil layer depth (m)
        self.root_beta = 0.98  # ext. roots (1^(cm-1); Gale & Grindal)
        self.Psie = -0.8e-3  # air entry point water potential (MPa)
        self.Ps = self.Psie  # initial soil water potential (MPa)
        self.hyds = 8.05e-6  # saturation hydraulic conductivity (m s-1)
        self.fc = 0.3  # field capacity (m3 m-3)
        self.theta_sat = 0.5  # soil moisture at saturation (m3 m-3)
        self.pwp = 0.05  # permanent wilting point (m3 m-3)
        self.bch = 4.  # Clapp and Hornberger index
        self.r_soil = 0.6  # resistance to soil evap (unitless)

        return
