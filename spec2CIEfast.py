#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spec2CIEfast.py - Convert spectra to color and brightness quickly.

This code is meant to test fast ways to go from a stars spectrum to its
brightness and color, in such a way that we can plot a starfield in a
reasonable way. We need to be able to do this very quickly, especially if we
need to deal with up to a million stars a more.
"""

__author__ = "Seth Bruzewski"
__email__ = "bruzewskis@gmail.com"
__created__ = "2022-04-06"
__modified__ = "2023-04-06"

import numpy as np
from scipy.constants import h, c, k
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import skimage


def planck(lam, T):
    # unit = W/m^3/sr
    B = 2*h*c**2/lam**5 / (np.exp(h*c/(k*lam*T))-1)
    return B


def planck_rel(lam, T, D):
    return planck(lam, T*D)/D**2


def dopp(beta, theta):
    costheta = (np.cos(theta)+beta)/(1+beta*np.cos(theta))
    D = np.sqrt(1-beta**2)/(1-beta*costheta)
    return D


def spectralradiosity(lam, mass, distance=3.09e17, D=1):
    # unit = W/m^2/m, defaults to Sun at 10 pc

    # Calculate properties from mass
    rad = mass**0.57 if mass > 1 else mass**0.8
    lum = mass**3.88
    temp = lum**(1/4) * rad**(-1/2) * 5780

    # Other stuff
    solid_angle = np.pi*(rad/distance)**2  # Taylor expansion 2pi(1-sqrt(1-(r/d)^2))
    sr = solid_angle * planck_rel(lam, temp, D)
    return sr


def g(x, amp, mu, sig1, sig2):
    res = amp * np.exp(-0.5 * (x-mu)**2 / np.piecewise(x, [x<mu, x>=mu], [sig1, sig2])**2)
    return res


def XYZgen(TD, parms):

    specrad = lambda lam:  2*h*c**2/lam**5 / (np.exp(h*c/(k*lam*TD))-1)
    cbar = lambda lam: np.sum([g(lam, *p) for p in parms], axis=0)
    Cfunc = lambda lam: specrad(lam*1e-9) * cbar(lam)

    ll = np.arange(380, 780, 0.1)
    C = quad(Cfunc, 380, 780)[0]

    return max([C, 1e-30])


def genXYZfile(DT):

    xparms = [[+1.056, 599.8, 37.9, 31.0],
              [+0.362, 442.0, 16.0, 26.7],
              [-0.065, 501.1, 20.4, 26.2]]
    yparms = [[+0.821, 568.8, 46.9, 40.5],
              [+0.286, 530.9, 16.3, 31.1]]
    zparms = [[+1.217, 437.0, 11.8, 36.0],
              [+0.681, 459.0, 26.0, 13.8]]

    XYZ = np.array([[ XYZgen(dt, p) for p in [xparms, yparms, zparms] ] for dt in DT ]).T
    DTXYZ = np.zeros((4, DT.size))
    DTXYZ[0] = DT
    DTXYZ[1:] = XYZ

    plt.plot(DT, XYZ[0], label='X')
    plt.plot(DT, XYZ[1], label='Y')
    plt.plot(DT, XYZ[2], label='Z')
    plt.legend()
    plt.loglog()
    plt.show()

    np.savetxt('XYZinterp.dat', DTXYZ)


def colors(mass, d, D=1):

    # Star properties
    radius = np.piecewise(mass, [mass < 1, mass >= 1],
                          [lambda m: m**0.57, lambda m: m**0.8])
    lum = mass**3.88
    temp = lum**(1/4) * radius**(-1/2) * 5780

    DTXYZ = np.loadtxt('XYZinterp.dat')
    XYZ = interp1d(DTXYZ[0], DTXYZ[1:])(temp*D)

    XYZn = XYZ / np.sum(XYZ, axis=0)
    lum = XYZ[1] * np.pi * ((radius*6.97e8)/(d*3.086e16))**2 / D**2

    M = np.array([[+3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, +1.8760108, +0.0415560],
                  [+0.0556434, -0.2040259, +1.0572252]])

    rgb = np.dot(M, XYZn)
    rgb = np.piecewise(rgb, [rgb<=0.0031308], [lambda c : 12.92*c,
                                               lambda c : 1.055*c**(1/2.4)-0.055])
    r,g,b = np.clip(rgb, 0, 1)

    return np.array([r, g, b, lum])


def show_stars(xs,ys,rgb,lum,ni=200,im_rad=3):

    plt.figure(figsize=(10, 10), dpi=192)

    x = np.linspace(-im_rad, im_rad, ni)
    X, Y = np.meshgrid(x, x)
    im = np.random.uniform(0, 0.03, (ni, ni, 3))

    sig = np.max([np.log10(lum*1e10), np.zeros(lum.size)], axis=0) * 4e-3
    rsqr = (X[..., None]-xs)**2 + (Y[..., None]-ys)**2
    g = np.exp(-rsqr/(2*sig**2))
    grgb = g[..., None] * rgb.T
    im += np.sum(grgb, axis=2)

    plt.imshow(im, extent=[-13, 13, -13, 13], origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/Users/bruzewskis/Dropbox/fakecluster.png', bbox_inches='tight')
    plt.show()


def glob_cluster_proto():

    np.random.seed(1337)
    ns = 400
    doppler = dopp(0., 0)
    d_cluster = 800  # pc
    half_light = 0.6  # pc
    tube_size = 3

    xyz = np.random.normal(0, half_light, (3, ns))
    xyz[2] += d_cluster

    nb = int(tube_size**2 * d_cluster * 2 * 0.14)
    xyzb = np.random.uniform(-tube_size, tube_size, ((3, nb)))
    xyzb[2] *= d_cluster/tube_size
    xyzb[2] += d_cluster
    xyz = np.column_stack([xyz, xyzb])

    masses = 10**(-0.5+np.abs(np.random.normal(0, 0.6, np.size(xyz, 1))))
    rgbY = colors(masses, xyz[2], doppler)
    rgba = rgbY.copy()
    rgba[3] = 1

    show_stars(xyz[0], xyz[1], rgbY[:3], rgbY[3])


def starim(X, Y, x0, y0, T, L, R, dist, DTXYZ, D):

    rsa = np.pi * ((R*6.97e8)/(dist*3.086e16))**2 / D**2
    rXYZ = interp1d(DTXYZ[0], DTXYZ[1:])(T*D) * rsa
    rXYZn = rXYZ / np.sum(rXYZ)

    ramp = np.sqrt(rXYZ[1])
    sig = 5e-2
    rsqr = (X-x0)**2 + (Y-y0)**2
    g = ramp*np.exp(-rsqr/(2*sig**2))
    rgXYZ = g[..., None] * rXYZn
    return rgXYZ


def stargrid(beta, ang):

    D = dopp(beta, ang)
    print('Doppler Factor:', D)
    n = 6
    masses = np.repeat(np.logspace(-1.5, 1, n), n)
    distances = np.tile(np.logspace(-2, 3, n), n)

    # Star properties
    radius = masses**np.piecewise(masses, [masses<1, masses>=1], [0.57, 0.8])
    lum = masses**3.88
    temp = lum**(1/4) * radius**(-1/2) * 5780
    solid_angle = np.pi * ((radius*6.97e8)/(distances*3.086e16))**2 / D**2

    DTXYZ = np.loadtxt('XYZinterp.dat')
    XYZ = interp1d(DTXYZ[0], DTXYZ[1:])(temp*D) * solid_angle
    XYZn = XYZ / np.sum(XYZ, axis=0)
    lum = XYZ[1]

    ind = np.arange(masses.size)
    off = (n-1)/2
    xs = (ind%n/off - 1) * 2.25
    ys = (ind//n/off - 1) * 2.25

    ni = 1000
    x = np.linspace(-3, 4, ni)
    y = np.linspace(-3, 3, ni)
    X, Y = np.meshgrid(x, y)
    im = np.zeros((ni, ni, 3))

    amp = np.sqrt(lum)
    sig = 5e-2
    rsqr = (X[...,None]-xs)**2 + (Y[...,None]-ys)**2
    g = amp*np.exp(-rsqr/(2*sig**2))
    gXYZ = g[...,None] * XYZn.T
    im += np.sum(gXYZ, axis=2)

    # Arcturus
    im += starim(X, Y, 3.15, 2.25, 4286, 170, 25.4, 11.26, DTXYZ, D)
    im += starim(X, Y, 3.15, 1.35, 12100, 121000, 78.9, 264, DTXYZ, D)
    im += starim(X, Y, 3.15, 0.45, 3600, 126000, 764, 168, DTXYZ, D)
    im += starim(X, Y, 3.15, -0.45, 3490, 270000, 1420, 1170, DTXYZ, D)
    im += starim(X, Y, 3.15, -1.35, 4.5e5, 2.2e6, 24, 10, DTXYZ, D)
    im += starim(X, Y, 3.15, -2.25, 3956, 295000, 1158, 10, DTXYZ, D)

    rgb = skimage.color.xyz2rgb(im)

    plt.figure(figsize=(12, 10), dpi=1920/20)
    # plt.subplot2grid((2,2),(0,0))
    plt.imshow(rgb, extent=[-3, 4, -3, 3], origin='lower')
    for i in range(len(masses)):
        label = f'M={masses[i]:.2f}\nD={distances[i]:.2f}'
        plt.text(xs[i], ys[i]-0.3, label, ha='center', va='top', c='w')

    plt.text(3.15, 2.25-0.3, 'Arcturus', ha='center', va='top', c='w')
    plt.text(3.15, 1.35-0.3, 'Deneb', ha='center', va='top', c='w')
    plt.text(3.15, 0.45-0.3, 'Betelgeuse', ha='center', va='top', c='w')
    plt.text(3.15, -0.45-0.3, 'VY Canis Majoris', ha='center', va='top', c='w')
    plt.text(3.15, -1.35-0.3, 'HD 5980A\nat 10pc', ha='center', va='top', c='w')
    plt.text(3.15, -2.25-0.3, 'RW Cephei\nat 10pc', ha='center', va='top', c='w')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig('/Users/bruzewskis/Dropbox/stars.png')

    return rgb


if __name__=='__main__':
    redo = False
    if redo:
        dts = np.logspace(-1, 8, 500)
        genXYZfile(dts)

    glob_cluster_proto()
    # test = stargrid(0, np.deg2rad(0))
