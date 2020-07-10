#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:22:48 2020

@author: bruzewskis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import gammainc
from scipy.integrate import quad
from matplotlib.colors import LogNorm
from colour_system import cs_hdtv as cs
import imageio
import os

# Set up space
side = 100 # pc

# Build mass function
mm = np.logspace(-2,2)
massfunc = lambda m : 10**(-0.35*(np.log10(m)+2)**2.3)

z = []
for i in range(len(mm)-1):
    ni = side*side*quad(massfunc, mm[i], mm[i+1])[0]
    z.append(ni)
nz = np.array(z)/sum(z)

# Draw masses
number = int(0.14*side*side)
masses = np.random.choice(mm[:-1], number, p=nz)

# Get luminosities
lums = np.zeros_like(masses)
for i in range(len(masses)):
    m = masses[i]
    if m<0.43:
        lums[i] = 0.23*m**2.3
    elif m<2:
        lums[i] = m**4
    elif m<55:
        lums[i] = 1.4*m**3.5
    else:
        lums[i] = 32000*m
        
# temps
radius = masses
temps = ((lums*3.8e26) / (5.7e-8 * (4 * np.pi * (radius*7e8)**2)))**(1/4)

# spatial
ct = np.full(number, 0)
x = np.random.uniform(-side/2, side/2, number)
y = np.random.uniform(-side/2, side/2, number)
z = np.random.uniform(-side/2, side/2, number)
f = 3.8e26 * lums / (4 * np.pi * 3e16*(x**2+y**2) )


##### RUN SIMS #####

# colors
def planck(lam, T, bp):
    c = 3e8
    h = 6.6e-34
    k = 1.38e-23
    lam_m = lam*np.sqrt((1+bp)/(1-bp)) / 1.e9
    fac = h*c/lam_m/k/T
    B = 2*h*c**2/lam_m**5 / (np.exp(fac) - 1)
    return B

def runsim(ct,x,y,z,lums,beta,n):
    
    gamma = 1/np.sqrt(1-beta**2)

    lam = np.arange(380,781,5, dtype=float)
    
    ctp = gamma * (ct - beta * 3e16 * z)
    xp = x
    yp = y
    zp = gamma * (z - beta * ct)
    f = 3.8e26 * lums / (4 * np.pi * 3e16*(xp**2+yp**2) )
    
    # colors
    c = np.zeros((number,4))
    for i in range(number):
        bp = np.cos(np.arctan2(np.sqrt(xp[i]**2+yp[i]**2), z[i])) * beta
        spec = planck(lam, temps[i], bp)
        rgb = cs.spec_to_rgb(spec)
        c[i,:3] = rgb
        
    # Add to colors
    logf = np.log(f/min(f))
    c[:,3] = logf/max(logf)
    
    
    thetap = np.arctan2(yp,xp)
    phip = -np.arctan2(np.sqrt(xp**2+yp**2),zp) + np.pi/2
    
    plt.figure(figsize=(12,6))
    plt.subplot(projection='mollweide')
    plt.scatter(thetap,phip, color=c, s=5)
    plt.tight_layout()
    plt.grid()
    plt.title('β={0:7.6f}'.format(beta))
    plt.savefig('lightspeedframes/frame{0:04d}.png'.format(n), dpi=150)
    plt.show()

frames = 15*30
betas = np.tanh(np.linspace(0,3/2*np.pi, frames))
plt.plot(betas)

for i in range(len(betas)):
    runsim(ct,x,y,z,lums,betas[i],i)
    
with imageio.get_writer('zoomtolight.gif', mode='I', duration=15/frames) as writer:
    for filename in sorted(os.listdir('lightspeedframes')):
        image = imageio.imread(os.path.join('lightspeedframes',filename))
        writer.append_data(image)