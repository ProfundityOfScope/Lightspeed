#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
starcylinder.py - An attempt to examine the distribution of stars we need.

Olber's Paradox can't be that hard right? Given a choice of IMF, there should
be some spatial distribution we can observe, as a given mass of star will have
some max distance it can be seen at. As you go further out, you need bigger
stars, but how does the odds of finding a bigger star go? More volume to spawn
stars, but there's still a limit to how big a star you can spawn in the Milky
Way.

This little side project can inform the distribution of stars we spawn in for
our lightspeed code to play with.
"""

__author__ = "Seth Bruzewski"
__email__ = "bruzewskis@gmail.com"
__created__ = "2022-04-15"
__modified__ = "2023-04-06"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import distributions


class MassFunction(object):
    """
    Generic Mass Function class.

    (this is mostly meant to be subclassed by other functions, not used itself)
    """

    def __init__(self, mmin=None, mmax=None):
        self._mmin = self.default_mmin if mmin is None else mmin
        self._mmax = self.default_mmax if mmax is None else mmax

    @property
    def mmin(self):
        return self._mmin

    @property
    def mmax(self):
        return self._mmax


class ChabrierPowerLaw(MassFunction):
    default_mmin = 1e-1
    default_mmax = 1e2

    def __init__(self,
                 lognormal_center=0.22,
                 lognormal_width=0.57*np.log(10),
                 mmin=default_mmin,
                 mmax=default_mmax,
                 alpha=2.3,
                 mmid=1):
        """
        From Equation 18 of Chabrier 2003
        https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract
        Parameters
        ----------
        lognormal_center : float
        lognormal_width : float
            The lognormal width.  Scipy.stats.lognorm uses log_n,
            so we need to scale this b/c Chabrier uses log_10
        mmin : float
        mmax : float
        alpha : float
            The high-mass power-law slope
        mmid : float
            The mass to transition from lognormal to power-law
        Notes
        -----
        A previous version of this function used sigma=0.55,
        center=0.2, and alpha=2.35, which come from McKee & Offner 2010
        (https://ui.adsabs.harvard.edu/abs/2010ApJ...716..167M/abstract)
        but those exact numbers don't appear in Chabrier 2005
        """
        # The numbers are from Eqn 3 of
        # https://ui.adsabs.harvard.edu/abs/2005ASSL..327...41C/abstract
        # There is no equation 3 in that paper, though?
        # importantly lognormal center is exp(M) where M is mean of ln(mass)
        # normal distribution
        super().__init__(mmin=mmin, mmax=mmax)
        self._mmid = mmid
        if self.mmax <= self._mmid:
            raise ValueError("The Chabrier Mass Function does not support "
                             "mmax <= mmid")
        self._alpha = alpha
        self._lognormal_width = lognormal_width
        self._lognormal_center = lognormal_center
        self.distr = distributions.CompositeDistribution([
            distributions.TruncatedLogNormal(self._lognormal_center,
                                             self._lognormal_width,
                                             self.mmin,
                                             self._mmid),
            distributions.PowerLaw(-self._alpha, self._mmid, self.mmax)
        ])

    def __call__(self, x, integral_form=False, **kw):
        if integral_form:
            return self.distr.cdf(x)
        else:
            return self.distr.pdf(x)


if __name__ == '__main__':
    n = 100000
    mfc = ChabrierPowerLaw()
    masses = mfc.distr.rvs(n)

    lum = masses**3.88 * 4e26  # W
    lum = np.piecewise(masses,
                       [masses < 0.43,
                        np.logical_and(0.43 < masses, masses < 2),
                        np.logical_and(2 < masses, masses < 55),
                        masses > 55],
                       [lambda m: 0.23*m**2.3,
                        lambda m: m**4,
                        lambda m: 1.4*m**3.5,
                        lambda m: 32000*m]) * 4e26
    min_flux = 4e24 / (4*np.pi*3e17**2)
    maxdist = 100  # np.sqrt(lum /(4*np.pi*min_flux))/3e16
    print(n/(2*np.pi*maxdist**3))

    radius = maxdist * np.sqrt(np.random.random(masses.size))
    theta = np.random.uniform(-np.pi, np.pi, masses.size)
    z = np.random.uniform(-10, 10, masses.size)
