#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lightspeed.py - Prototype code which demonstrates the effect.

Generates an animation which approximates the look of the effect. This code
only uses a collection of sun-like stars, randomly distributed, and does not
adequately describe the change in color and brightness, although it serves as
a decent approximation. Roughly follows this derivation:
https://arxiv.org/pdf/physics/0510113.pdf
"""

__author__ = "Seth Bruzewski"
__email__ = "bruzewskis@gmail.com"
__created__ = "2020-07-08"
__modified__ = "2023-04-06"

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k,h,c
from colour_system import cs_hdtv as cs
import imageio
from tqdm import trange, tqdm

def random_sky(n):
    '''
    Generatea a random distribution on the sky
    
    n (int) - Number to generate
    '''
    theta = np.random.uniform(-np.pi,np.pi,n)
    phi = np.arcsin(np.random.uniform(-1,1,n))
    
    return theta, phi

def zspec(lam, t0, b):
    '''
    Calculates redshifted spectra. Note that the input beta values must be the
    component of the beta vector parallel to the vector toward the star. 
    
    lam (float) - wavelength in meters
    t0 (float) - blackbody temperature in Kelvin
    b (float) - parallel component of beta vector
    '''
    num = (2*h*c**2/lam**5) * (1-b**2)**(3/2)
    den = (1-b)**3 * (np.exp( h*c/(k*lam*t0) * (1-b)/np.sqrt(1-b**2) )-1)
    return num/den
    
def gofast(b_lon, b_lat, beta, nstars=1000, seed=42):
    '''
    Simulate the instantaneous sky for an observer moving with a velocity
    magnitude of beta in the direction (b_lon, b_lat). The field is composed
    of nstars exactly replicas of our sun because realistically populating a 
    galaxy isn't fun. You can also change the seed.
    
    b_lon (float) - The longitude of the beta vector.
    b_lat (float) - The latitude of the beta vector.
    beta (float) - Magnitude of velocity, -1 to 1.
    nstars (int) - Number of stars to draw.
    seed (int) - Random seed, stays the same for animations
    '''
    
    # Beta vector
    np.random.seed(seed)
    bv = np.array([np.cos(b_lat) * np.cos(b_lon),
                   np.cos(b_lat) * np.sin(b_lon),
                   np.sin(b_lat)])
    
    # Draw out sky
    theta, phi = random_sky(nstars)
    xyz = np.array([np.cos(phi) * np.cos(theta),
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi)])
    gamma = 1/np.sqrt(1-beta**2)
    
    # Rotate to axis
    rz = lambda lon : np.array([[np.cos(lon),   -np.sin(lon),  0],
                                 [np.sin(lon),  np.cos(lon),   0],
                                 [0,              0,              1]])
    ry = lambda lat : np.array([[np.cos(lat),    0,  np.sin(lat)],
                                [0,             1,  0],
                                [-np.sin(lat),  0,  np.cos(lat)]])
    rotation = ry(b_lat).dot(rz(-b_lon))
    rotation_inv = rz(b_lon).dot(ry(-b_lat))
    rotated = rotation.dot(xyz)
    
    # Transform to ellipse
    transform = np.array([[gamma,0,0],[0,1,0],[0,0,1]])
    offset = np.array([[gamma*beta],[0],[0]])
    
    scaled = transform.dot(rotated) + offset
    
    # De-rotate
    nxyz = rotation_inv.dot(scaled)
    
    # Go back to spherical coords
    lon = np.arctan2(nxyz[1], nxyz[0])
    lat = np.arctan2(nxyz[2], np.sqrt(nxyz[0]**2+nxyz[1]**2))
    
    # Get the distances
    scale = np.linalg.norm(nxyz, axis=0)
    
    # Note: the 60 here comes from the fact that our sun would have an
    # apparent magnitude of about 8 at that distance, since all the stars
    # are clones of the sun, we can reasonably assume thats as far as one
    # could see them out to.
    distances = 60*3e16*np.random.random(nstars)**(1/3) * scale # meters
    
    # Figure out colors
    rgba = np.ones((nstars,4))
    b_par = bv.dot(xyz) * beta
    lam = np.arange(380,781,5, dtype=float) * 1e-9
    get_rgb = lambda b : cs.spec_to_rgb(zspec(lam,6000,b))
    for i in range(nstars):
        rgb = get_rgb(b_par[i])
        rgba[i,:3] = rgb
        
    # Figure out brightness
    flux = 3.8e26 / ( 4*np.pi * (distances)**2)
    logf = np.log(flux/min(flux)) # My "human-eye-like" scaling
    rgba[:,3] = logf/max(logf)
    
    return lon, lat, rgba
    
def animated(bmax, tmax, fps=30, save=None):
    '''
    Given a max velocity and how long you want to take to get there, this
    code will animated a movie for you, a high quality one
    
    bmax (float) - Target beta parameter (v/c). Between -1 and 1.
    tmax (float) - Time to reach bmax in seconds.
    fps (int) - Frames per second, duh.
    save (str) - Name to write to.
    '''
    
    accel = np.arctanh(bmax)*c/(2*np.pi*tmax)
    
    plt.style.use('dark_background')
    time = np.linspace(0,tmax,int(tmax*fps))
    betas = np.tanh(2*np.pi*accel*time/3e8)
    print('Accelerating at', format(accel, '1.2e'), 'm/s/s')
    
    # Loop for images
    ims = []
    for i in trange(len(time)):
        # Set up plots
        fig = plt.figure(figsize=(16,9), dpi=240)
        
        # First plot
        ax1 = fig.add_axes([0.01, 0.01, 0.98, 0.98], projection='mollweide')
        lon, lat, rgba = gofast(np.deg2rad(90),np.deg2rad(0),betas[i])
        plt.scatter(lon, lat, color=rgba, s=10, marker='.')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('Sky View (Mollweide)')
        
        # Speed plot
        ax2 = fig.add_axes([0.04, 0.093, 0.25, 0.25])
        plt.plot(time,betas)
        plt.scatter(time[i],betas[i])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [c]')
        plt.ylim(-0.1,1.1)
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ims.append(image)
        plt.close()
    
    if not save is None:
        kwargs = {'fps':fps, 'quality':9, 'codec':'h264'}
        with imageio.get_writer(save, **kwargs) as writer:
            for im in tqdm(ims):
                writer.append_data(im)
    
    return None
    
# Setup
if __name__=='__main__':
    # This line will run an animation script
    #animated(0.999, 30, 60, 'test.mp4')
    
    plt.style.use('dark_background')
    plt.figure(figsize=(16,9), dpi=240)
    plt.subplot(projection='mollweide')
    lon, lat, rgba = gofast(np.deg2rad(45), np.deg2rad(0), 0.8, seed=1)
    plt.scatter(lon,lat, color=rgba, edgecolor='none', s=20, zorder=10)
    plt.xticks(np.linspace(-np.pi,np.pi,13),[])
    plt.yticks(np.linspace(-np.pi/2,np.pi/2,13),[])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig('thumbnail.png')











