import argparse
import math
import os
import sys
import pandas as pd

import astropy.io.fits as pf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from matplotlib import colors
from scipy import asarray as ar
from scipy import exp, stats
from scipy.optimize import curve_fit
from scipy.optimize import Bounds
from matplotlib.figure import Figure


def get_modulation_factor_stokes(theta_array):
    """Function to estimate the modulation factor according to the method described in the Standford paper"""
    w=np.ones(len(theta_array))
    _theta=theta_array
    N = sum(w)
    U=2*sum(w*np.sin(2*_theta))
    Q=2*sum(w*np.cos(2*_theta))
    Q_n = Q/N
    U_n = U/N
    mu=math.sqrt(Q_n**2+U_n**2)
    pred_phi=0.5*np.arctan2(U,Q)
    sigma_Q_n = np.sqrt((1/(N-1))*(2/mu**2-Q_n**2))
    sigma_U_n = np.sqrt((1/(N-1))*(2/mu**2-U_n**2))
    #print(U/N, sigma_U)
    err_mu = np.sqrt((Q_n*sigma_Q_n/(np.sqrt(Q_n**2+U_n**2)))**2+(U_n*sigma_U_n/(np.sqrt(Q_n**2+U_n**2)))**2)
    return mu, pred_phi, err_mu

#####################################################################    
#####################################################################

def color_plot(color):
    if color=='red':
        r = 1
        g = 0
        b = 0
    if color=='blue':
        r = 0
        g = 0
        b = 1
    if color=='green':
        r = 0
        g = 0.5
        b = 0
    if color=='black':
        r = 0
        g = 0
        b = 0
    if color=='cyan':
        r = 0
        g = 1
        b = 1
    if color=='magenta':
        r = 1
        g = 0
        b = 1
    if color=='purple':
        r = 0.5
        g = 0
        b = 0.5
    if color=='orange':
        r = 1
        g = 0.5
        b = 0
    return r, g, b

#####################################################################    
#####################################################################

def readfitsfile(file_path):
    """
        Function reads fits file
    """
    data_f = pf.open(file_path)
    data_f.info()
    events = data_f['EVENTS'].data
    
    return events

#####################################################################    
#####################################################################
"""
    gauss function to fit spectrum peaks
"""
#def gauss(x, H, A, x0, sigma):
#	return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss(x,  A, x0, sigma):
	return  A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


    

#####################################################################    
#####################################################################
"""
    modulation function to fit the phi angle from GPD data
"""
def modulation_function(x, A, B, phi):
	return A + B*(np.cos((x-phi)))**2

#####################################################################    
#####################################################################

def spectrum(pha, binning, n_sigma_peak, fit=bool, title_label=str, color=str):
    """
        function to create and plot the spectrum from an array of events (pha)
        with len(pha) = # of events
        binning is the binning od the histo of the spectrum
        n_sigma_peak:
            the fit is performed from peak_center-n_sigma_peak*0.10*peak_center
        fit = bool, True if you want to fit
        title_label = title of the spectrum
        color = histo color (filled too)
    """
    r, g, b = color_plot(color)
    #fig_spectrum, ax_spectrum = plt.subplots(figsize =(7, 7),label=title_label)
    fig_spectrum, ax_spectrum = plt.subplots(figsize =(7, 7))   
    spettro = ax_spectrum.hist(pha,binning,facecolor=(r,g,b,0.2),edgecolor=(r,g,b,1),histtype='step',label='spectrum', fill=True)
    data = spettro[0]
    histo_bins = spettro[1]
    bins_centers = np.array([0.5 * (histo_bins[i] + histo_bins[i+1]) for i in range(len(histo_bins)-1)])
    x_max = np.where(data==np.max(data))[0][0]
    lim_inf = bins_centers[x_max]-n_sigma_peak*0.5*bins_centers[x_max]
    lim_sup = bins_centers[x_max]+n_sigma_peak*0.5*bins_centers[x_max]
    
    x_peak = np.where((bins_centers>lim_inf) & (bins_centers<lim_sup))[0]
    ax_spectrum.set_xlabel('ADC counts')
    ax_spectrum.set_ylabel('Entries')
    ax_spectrum.set_title(title_label)
    ax_spectrum.set_xlim(left=0)
    ax_spectrum.set_xlim(right=3*np.max(bins_centers[x_max]))
    ax_spectrum.grid()
    ax_spectrum.legend()
    if fit==True:
        par_spectrum, cov_spectrum = curve_fit(gauss, xdata=bins_centers[x_peak], ydata=data[x_peak], p0=[ np.max(data), bins_centers[x_max], 0.2*bins_centers[x_max] ])
        mean=par_spectrum[1]
        sigma=par_spectrum[2]

        #iterative fit:
        for i in range(0,3):
            x_peak = np.where((bins_centers>mean-n_sigma_peak*sigma) & (bins_centers<mean+n_sigma_peak*sigma))[0]
            par_spectrum, cov_spectrum = curve_fit(gauss, xdata=bins_centers[x_peak], ydata=data[x_peak], p0=[ par_spectrum[0], par_spectrum[1],   par_spectrum[2]  ])
            mean=par_spectrum[1]
            sigma=par_spectrum[2]
            print("iter= ",i,"meam= ",mean,"+-", np.sqrt(cov_spectrum[1][1]), " sigma=",sigma,"+-",cov_spectrum[2][2])
           


        chi2=np.sum(((gauss(bins_centers[x_peak],par_spectrum[0],mean,sigma)-data[x_peak])**2)/data[x_peak]) 
        dof=len(data[x_peak])-3
        print("chi2=",chi2," dof=",dof," chi2/dof=",chi2/dof)
        
        x = np.linspace(mean-sigma*n_sigma_peak,mean+sigma*n_sigma_peak, 1000)
        ax_spectrum.plot(x,gauss(x,par_spectrum[0],par_spectrum[1],par_spectrum[2]),linestyle='-',color='red',linewidth=2)
        ax_spectrum.legend(('spectrum','\n'+'fit'+'\n'+
            r'$\mu$='+"%.2f" % par_spectrum[1]+'+-'+"%.2f" % np.sqrt(cov_spectrum[1][1])+'\n'+
            r'$\sigma$='+"%.2f" % par_spectrum[2]+'+-'+"%.2f" % np.sqrt(cov_spectrum[2][2])),loc='upper right',shadow=True)
    else:
        par_spectrum = 'fit = False'
        cov_spectrum = 'fit = False'
    return data, histo_bins, par_spectrum, cov_spectrum, ax_spectrum

#####################################################################    
#####################################################################

def modulation(phi, bins, fit=bool, title_label=str, color=str):
    """
        modulation analysis from the phi data (phi) 
        from the GPD, binned with bins
        fit = bool, True if you want to fit
        title_label = title of the spectrum
        color = histo color (filled too)
    """
    r, g, b = color_plot(color)
    x_phi = np.linspace(-np.pi, np.pi, 1000)
    #fig_mod, ax_mod = plt.subplots(figsize=(10,7),label=title_label)
    fig_mod, ax_mod = plt.subplots(figsize=(10,7))
   
    histo_phi = plt.hist(phi, bins, facecolor=(r,g,b,0.2),edgecolor=(r,g,b,1),histtype='step',label=r'$\Phi$', fill=True)
    histo_phi_data = histo_phi[0]
    histo_phi_bins = histo_phi[1]
    bins_centers_phi = np.array([0.5 * (histo_phi_bins[i] + histo_phi_bins[i+1]) for i in range(len(histo_phi_bins)-1)])
    ax_mod.set_title(title_label)
    ax_mod.set_ylabel(r'$N(\Phi)$')
    ax_mod.set_xlabel(r'$\Phi$')
    #ax_mod.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'$-\pi$', r'$-\pi/2$','0',r'$\pi/2$', r'$\pi$'])
    ax_mod.grid()
    ax_mod.legend()
    if fit==True:
        par_phi, cov_phi = curve_fit(modulation_function, xdata=bins_centers_phi, ydata=histo_phi_data)
        A = par_phi[0]
        B = par_phi[1]
        #if A<0 or B<0:
        #    print('!!!')
        phi0 = par_phi[2]
        A_err = np.sqrt(cov_phi[0][0])
        B_err = np.sqrt(cov_phi[1][1])
        phi0_err = np.sqrt(cov_phi[2][2])
        mu = np.abs(B/(2*A+B)*100.)
        mu_err = 2*np.sqrt((B_err*A/(2*A+B)**2)**2+(A_err*B/(2*A+B)**2)**2)*100.

        ax_mod.plot(x_phi,modulation_function(x_phi,par_phi[0],par_phi[1],par_phi[2]),linestyle='-',color='red',linewidth=2)
        phase = np.degrees(phi0)%360.
        if phase>180:
            phase=phase-360
        if phase<180:
            phase=phase-180
        phase_err = np.degrees(phi0_err)
        ax_mod.legend((r'$N(\Phi)$',r'$N(\Phi)=A+B\cdot cos^2(\Phi-\Phi_0)$'+'\n'+
            r'$\Phi_0$='+"%.2f" % phase+'+-'+"%.2f" % phase_err + '°' +'\n'+
            r'$\mu$='+"%.1f" % mu + r'$\pm$' + "%.1f" % mu_err + '%'),loc='lower center')

        chi2=np.sum(((modulation_function( bins_centers_phi,A,B,phi0)-histo_phi_data)**2)/histo_phi_data) 
        dof=len(histo_phi_data)-3

        print ("A=",A, "+-",A_err)
        print ("B=",B, "+-",B_err)
        print ("phi0=",phi0, "+-",phi0_err)
        print("chi2=",chi2," dof=",dof," chi2/dof=",chi2/dof)
        
        
    else:
        phase = 'fit = False'
        phase_err = 'fit = False'
        mu = 'fit = False'
        mu = 'fit = False'

    return histo_phi_data, histo_phi_bins, phase, phase_err, mu, mu_err, ax_mod

#####################################################################    
#####################################################################

def absorption_map(x_abs, y_abs, bins):
    """
        creation and plot of the map of the 
        absorption points (x_abs,y_abs)
        binned with bins
    """
    fig_abs_map, ax_abs_map = plt.subplots(figsize=(7,7))
    ax_abs_map.axis('square')
    #counts_abs_map, xedges_abs_map, yedges_abs_map, im_abs_map = ax_abs_map.hist2d(x_abs, y_abs,bins,weights=[1/len(x_abs)]*len(x_abs),range=[[-8,8],[-8,8]], cmap=mpl.cm.hot)
    counts_abs_map, xedges_abs_map, yedges_abs_map, im_abs_map = ax_abs_map.hist2d(x_abs, y_abs,bins,range=[[-8,8],[-8,8]], cmap=mpl.cm.hot)
    
    x_bar_bin = np.array([0.5 * (xedges_abs_map[i] + xedges_abs_map[i+1]) for i in range(len(xedges_abs_map)-1)])
    y_bar_bin = np.array([0.5 * (yedges_abs_map[i] + yedges_abs_map[i+1]) for i in range(len(yedges_abs_map)-1)])
    ax_abs_map.set_title('abs')
    ax_abs_map.set_xlabel('mm')
    ax_abs_map.set_ylabel('mm')
    fig_abs_map.colorbar(im_abs_map, ax=ax_abs_map)
    return counts_abs_map, xedges_abs_map, yedges_abs_map, ax_abs_map, x_bar_bin, y_bar_bin

#####################################################################    
#####################################################################
    
def data_cut(old_data, data, cut_inf, cut_sup):
    """
        cut function:
        old_data is an array in whish every entry is an array to be cut
        data is the array which rules the cut: for example, a position cut on the y coordinate
        of the absorption point of the photon in the GPD requires data=y_abs
        cut_inf and cut_sup are the limits of the cut
        new_data is an array in which every entry is the correspondant old_data entry array but cut
    """
    mask_cut = np.where((data<cut_sup) & (data>cut_inf))
    new_data = []
    for i in range(len(old_data)):
        new_data.append(np.zeros(len(mask_cut)))
        new_data[i] = old_data[i][mask_cut]
    return new_data
