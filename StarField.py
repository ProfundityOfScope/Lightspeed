#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StarField.py - Generate a somewhat realistic field of stars.

Has a wrapper class that lets you quickly handle the generation and
manipulation of a large number of stars, mostly for convenience.
"""

__author__ = "Seth Bruzewski"
__email__ = "bruzewskis@gmail.com"
__created__ = "2022-04-15"
__modified__ = "2023-04-06"

import numpy as np
from scipy.constants import h, c, k
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import distributions
from scipy.interpolate import interp1d
from skimage.color import xyz2rgb


class MassFunction(object):
    """
    Generic Mass Function class
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
    default_mmin = 1e-2
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
        # importantly the lognormal center is the exp(M) where M is the mean of ln(mass)
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


class StarField():

    def __init__(self, n=1000, positions=None, seed=None):
        
        self.num = n
        np.random.seed(seed)

        # Generate masses
        mfc = ChabrierPowerLaw()
        self.masses = mfc.distr.rvs(n)

        # Figure out other properties from mass
        self.radii = np.array(list(map(lambda m: m**0.57 if m>1 else m**0.8, self.masses)))
        self.luminosities = self.masses**3.88
        self.temperatures = self.luminosities**(1/4) * self.radii**(-1/2) * 6000

        # This might need its own distribution method
        if positions is None:
            length = (n/0.14)**(1/3)  # This needs fixing
            self.positions = np.random.uniform(-length/2, length/2, (3, n))
        else:
            self.positions = positions
        self.distances = np.linalg.norm(self.positions, axis=0)
        self.x = self.positions[0]
        self.y = self.positions[1]
        self.z = self.positions[2]

        # Initialize relativity
        self._beta = 0
        
        # Initialize color
        self._colorfile = 'XYZinterp.dat'

    def __len__(self):
        return self.num

    @property
    def sky_positions(self):

        pitch = np.arctan2(np.sqrt(self.x**2+self.y**2), self.z)
        roll = np.arctan2(self.y, self.x)

        # costheta = (np.cos(theta)+beta)/(1+beta*np.cos(theta))
        cosp = np.cos(pitch)
        pitch_shifted = np.arccos((cosp+self.beta)/(1+self.beta*cosp))

        return pitch_shifted, roll

    @property
    def pitch(self):
        pitch0 = np.arctan2(np.sqrt(self.x**2+self.y**2), self.z)
        cosp = np.cos(pitch0)
        pitch = np.arccos((cosp+self.beta)/(1+self.beta*cosp))
        return pitch

    @property
    def roll(self):
        roll = np.arctan2(self.y, self.x)
        return roll

    @property
    def sky_positions_xyz(self):
        spx = np.sin(self.pitch) * np.cos(self.roll)
        spy = np.sin(self.pitch) * np.sin(self.roll)
        spz = np.cos(self.pitch)

        return spx, spy, spz

    def spectrum(self, lam):

        Tr = self.temperatures * self.doppler_factor

        # These just shape things correctly
        prefactor = np.divide.outer(2*h*c**3*self.doppler_factor, lam**5)
        lamTr = np.multiply.outer(Tr, lam)

        blackbody = prefactor / (np.exp(h*c/(k*lamTr))-1)

        return blackbody

    @property
    def brightnesses(self):
        return self.doppler_factor**2 * self.luminosities / (4*np.pi*self.distances**2)

    @property
    def colors(self):
        
        radi_meters = self.radii*6.97e8
        dist_meters = self.distances*3.086e16
        solid_angle = np.pi * ((radi_meters)/(dist_meters))**2
        
        dopp = self.doppler_factor
        DT = self.temperatures * dopp
        
        DTXYZ = np.loadtxt(self._colorfile)
        XYZ = interp1d(DTXYZ[0], DTXYZ[1:])(DT) * solid_angle / dopp**2
        
        return XYZ

    @property
    def doppler_factor(self):
        dopp = np.sqrt(1-self.beta**2)/(1-self.beta*np.cos(self.pitch))
        return dopp

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if not 0<=value<=1:
            raise ValueError('Velocity must be 0-1')
        self._beta = value


if __name__=='__main__':

    N = 100
    pos = np.random.normal(0, 0.1, (3,N))
    pos[2] += 10
    stars = StarField(N, positions=pos)
    test = stars.colors

    x = np.linspace(-0.05, 0.05, 201)
    y = np.linspace(-0.05, 0.05, 200)
    X,Y = np.meshgrid(x,y)
    Z = np.random.uniform(1e-30, 1e-32, (3,y.size,x.size))
    
    xs = np.arctan2(stars.x, stars.z)
    ys = np.arctan2(stars.y, stars.z)
    colors = stars.colors
    
    for i in range(len(stars)):
        r_sqr = (X-xs[i])**2 + (Y-ys[i])**2
        g = colors[:,i,None,None] * np.exp(-r_sqr/(2*(6e-4)**2))
        Z += g
        
    rgb = xyz2rgb(Z, channel_axis=0)
    rgb = np.sqrt(rgb)
    
    cmin = 1e-30
    cmax = 1e-13
    rgb[rgb>cmax] = cmax
    
    plt.figure(figsize=(10,10))
    plt.subplot2grid((2,2),(0,1))
    b = np.logspace(-50, 0)
    plt.hist(rgb[0].ravel(), histtype=u'step', bins=b, color='r')
    plt.hist(rgb[1].ravel(), histtype=u'step', bins=b, color='g')
    plt.hist(rgb[2].ravel(), histtype=u'step', bins=b, color='b')
    plt.axvline(cmin, c='k', ls='--')
    plt.axvline(cmax, c='k', ls='--')
    plt.xlim(1e-22, 1)
    plt.loglog()
    
    plt.subplot2grid((2,2),(0,0))
    rgb_im = np.transpose(rgb, (1,2,0))
    plt.imshow(rgb_im, origin='lower',
               extent=[-0.05, 0.05, -0.05, 0.05])
    
    plt.subplot2grid((2,2),(1,0))
    plt.scatter(xs, ys, c=np.sum(colors, axis=0), norm=LogNorm())
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    
    plt.subplot2grid((2,2),(1,1))
    plt.imshow(Z[1], origin='lower', norm=LogNorm())