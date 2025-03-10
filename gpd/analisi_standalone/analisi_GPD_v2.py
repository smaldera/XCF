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
from GPD_functions import spectrum, get_modulation_factor_stokes, \
    readfitsfile, modulation, absorption_map, data_cut

binning = 1000 #bins for the spectrum
bins_modulation = 100 #bins for the modulation
bins_map = 100 #bins for the maps
bins_projection = 100 #bins for the projection map
bins_c = 500
n_sigma = 1 #cut under the peak

#Define the Gaussian function
#def gauss(x, H, A, x0, sigma):
#	return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


#Define the modulation function
def modulation_function(x, A, B, phi):
	return A + B*(np.cos(x-phi))**2


parser = argparse.ArgumentParser(description='Descrizione')
parser.add_argument('filename', metavar='FILENAME', type=str, help='Recon fits file to analyze')

args = parser.parse_args()

filename = args.filename
#name = os.path.split(os.path.abspath(path))
name=os.path.dirname(filename)
events = readfitsfile(filename)

x_abs=events['ABSX']
y_abs=events['ABSY']
x_bar=events['BARX']
y_bar=events['BARY']
pha = events['PHA']
phi1 = events['DETPHI1']
phi2 = events['DETPHI2']
time=events['TIME']
num_clu=events['NUM_CLU']
livetime=events['LIVETIME']

print ("len(num_clu)",len(num_clu))
print ("len(livetime)",len(livetime))
print ("len(phi1)",len(phi1))
print ("len(x_abs)",len(x_abs))
print ("len(y_abs)",len(y_abs))


mask = np.where((pha>0) & (pha<60000))[0]      
mask_col = np.where((x_abs<=7) & (x_abs>=-7))[0]
mask_row = np.where((y_abs<=7) & (y_abs>=-7))[0]

base_cut= np.where((pha>0) & (pha<60000) & (x_abs<=7) & (x_abs>=-7) &(y_abs<=7) & (y_abs>=-7) &(livetime>15 )& (num_clu>0)   )[0]


#applico il base_cut a tutte le variabili
x_abs=x_abs[base_cut]
y_abs=y_abs[base_cut]
x_bar=x_bar[base_cut]
y_bar=y_bar[base_cut]
pha = pha[base_cut]
phi1 =phi1[base_cut]
phi2 =phi2[base_cut]
time=time[base_cut]
num_clu=num_clu[base_cut]
livetime=livetime[base_cut]
                 
                 
Dt = events['TIME'].max() - events['TIME'].min()
tot_events = len(pha)
###############################################################################################################################
"""
    plot the spectrum.
    The peak is fitted using a gaussian, 
    only in the region near the peak
"""
"""
    cut to avoid high energy counts
    limit 60000 ADC almost 15 keV, more than the max 
    available with the tube: 10 kV
"""

print('====================================================================================')
print('Number of events = ',tot_events)
print('On source time = ', Dt, ' s')

data, histo_bins, par_spectrum, cov_spectrum, ax_spectrum = spectrum(pha=pha,binning=binning,n_sigma_peak =1.5, fit=True, title_label='PHA spectrum', color='blue')
print('spectrum peak = ', float("%.2f" % (par_spectrum[1])), ' +- ', float("%.2f" % (np.sqrt(cov_spectrum[1][1]))))
print('spectrum sigma = ', float("%.2f" % (par_spectrum[2])), ' +- ', float("%.2f" % (np.sqrt(cov_spectrum[2][2]))))
print("resolution=   ",2.355*par_spectrum[2]/(par_spectrum[1]))
print('====================================================================================')
mean_fit=par_spectrum[1]
mean_fitErr=np.sqrt(cov_spectrum[1][1])
sigma_fit=par_spectrum[2]
sigma_fitErr=np.sqrt(cov_spectrum[2][2])


# make energy cut
n_sigma_cut=1.5
energy_cut=np.where( (pha>mean_fit-sigma_fit*n_sigma_cut)& (pha<mean_fit+sigma_fit*n_sigma_cut) )

x_abs=x_abs[energy_cut]
y_abs=y_abs[energy_cut]
x_bar=x_bar[energy_cut]
y_bar=y_bar[energy_cut]
pha = pha[energy_cut]
phi1 =phi1[energy_cut]
phi2 =phi2[energy_cut]
time=time[energy_cut]
num_clu=num_clu[energy_cut]
livetime=livetime[energy_cut]

# plot surviving events
plt.hist(pha,histo_bins,facecolor=(1,0,0,0.1),edgecolor=(1,0,0,0.1),histtype='step',label='selected events', fill=True)
plt.legend()



###############################################################################################################################
"""
    plot of phi1 and phi2 histograms as function of the phase
    the modulation function A + B*(np.cos(x-phi))**2 has been
    used to fit.
    The modulation factor is mu = B/(2A+B)
"""
print("fit modulation...")

histo_phi_data, histo_phi_bins, phase, phase_err, mu, mu_err, ax_mod = modulation(phi=phi2,bins=bins_modulation,fit=True, title_label='modulation', color='blue')

###############################################################################################################################
"""
    plot of the map of barycenters and 
    plot of the map of absorption points
    in the detector
"""
counts_abs_map, xedges_abs_map, yedges_abs_map, ax_abs_map, x_bar_bin, y_bar_bin = absorption_map(x_abs=x_abs,y_abs=y_abs,bins=bins_map)

#if cut_type!=None:
fig_counts_map=plt.figure(figsize =(14, 7))
ax1=plt.subplot(1,2,1,title="X projection")
ax1.hist(x_abs,bins=bins_projection,edgecolor='blue',facecolor=(0,0,1,0.2),fill=True,histtype='step')
ax1.set_xlabel('X GPD [mm] ')
ax1.set_ylabel('counts')
plt.grid()

ax2=plt.subplot(1,2,2,title="Y projection")
ax2.hist(y_abs,bins=bins_projection,edgecolor='blue',facecolor=(0,0,1,0.2),fill=True,histtype='step')
ax2.set_xlabel('Y GPD [mm] ')
ax2.set_ylabel('counts')
plt.grid()


mu_stokes, pred_phi, err_mu = get_modulation_factor_stokes(np.array(phi2))

print('====================================================================================')
print('mu from fit = ', float("%.2f" % mu), ' +- ', float("%.2f" % mu_err))
print('phi from fit = ', phase)
print('====================================================================================')
print('mu from stokes = ', float("%.2f" % (mu_stokes*100.)), ' +- ', float("%.2f" % (err_mu*100.)))
print('phi from stokes = ', pred_phi)
print('====================================================================================')




"""
    energy cut
    modulation plot and maps
"""


#x_abs_max = xedges_abs_map_pos_cut[index]
#y_abs_max = yedges_abs_map_pos_cut[index]
#print(x_abs_max, y_abs_max)






plt.show()
