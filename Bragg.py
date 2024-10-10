import numpy as np
import matplotlib.pyplot as plt
import sys
import astropy.io.fits as pf
import os
from scipy.optimize import curve_fit
import math
from itertools import chain
import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-e','--ELEMENT', type=str,  help='Element of the Energy desired', required=True)
parser.add_argument('-en','--ENERGY', type=float,  help='Energy [keV] to evaluate the angle', required=False, default=None)
parser.add_argument('-a','--ANGLE', type=float,  help='Angle to evaluate the energy', required=False, default=None)

args = parser.parse_args()
ELEMENT = args.ELEMENT
ENERGY = args.ENERGY
ANGLE = args.ANGLE


if ELEMENT=='Mo':
    n = 0
if ELEMENT=='Rh':
    n = 1
if ELEMENT=='Pd':
    n = 2
if ELEMENT=='Ti':
    n = 3
if ELEMENT=='Fe':
    n = 4
if ELEMENT=='Ni':
    n = 5

hc = 1.240e-9 # keV m

 # m
theta_Mo = 46.28
theta_Rh = 44.87
theta_Pd = 44.12
theta_Ti = 45.71
theta_Fe = 45.5
theta_Ni = 45.86

dd_Mo = 7.481e-10
dd_Rh = 6.532e-10
dd_Pd = 6.271e-10
dd_Ti = 3.840e-10
dd_Fe = 2.7142e-10
dd_Ni = 2.31e-10

E_Mo = 2.2932
E_Rh = 2.697
E_Pd = 2.839
E_Ti = 4.511
E_Fe = 6.4
E_Ni = 7.478

E = [E_Mo,E_Rh,E_Pd,E_Ti,E_Fe,E_Ni]
dd = [dd_Mo,dd_Rh,dd_Pd,dd_Ti,dd_Fe,dd_Ni]
theta = [theta_Mo,theta_Rh,theta_Pd,theta_Ti,theta_Fe,theta_Ni]

def Bragg(theta,dd,n):
    E = n*hc/(dd*np.sin(np.radians(theta)))
    return E

def Der_Bragg(theta,dd,n):
    der=n*hc/(dd*np.sin(np.radians(theta)))/np.tan(np.radians(theta))
    return der

def Angle(Energy,dd,n):
    angle = np.arcsin(n*hc/(dd*Energy))
    return np.degrees(angle)

x=np.linspace(0,180,1000)

Energies = Bragg(x,dd[n],1)
min = np.min(Energies)

if ENERGY is not None:
    theta_ = Angle(ENERGY,dd[n],1)
    print(f'Energy = {ENERGY} keV ===> Angle = {theta_} °')

if ANGLE is not None:
    EN_ = Bragg(ANGLE,dd[n],1)
    print(f'Angle = {ANGLE} ° ===> Energy = {EN_} keV ')

plt.figure()
plt.plot(x,Bragg(x,dd[n],1),marker='',label='n=1')
# plt.plot(x,Bragg(x,dd_Mo,2),marker='.',label='n=2')
plt.axhline(y=E[n], label=ELEMENT)
plt.axhline(y=min, label='min E = '+"%.3f"%min+' keV', linewidth=2, color='red')
plt.fill_between(x, y1=0, y2=min, alpha=0.5, color='gray')
if ENERGY is not None:
    plt.axhline(ENERGY,linestyle='--',color='green')
    plt.axvline(x=theta_-1,color='gray',linestyle='--')
    plt.axvline(x=theta_+1,color='gray',linestyle='--')
    en_inf = Bragg(theta_-1,dd[n],1)
    en_sup = Bragg(theta_+1,dd[n],1)
    plt.axhline(y=en_inf,color='gray',linestyle='--')
    plt.axhline(y=en_sup,color='gray',linestyle='--')
    print(f'delta Energy = {(en_inf-en_sup)*1000.} eV')
plt.ylim([0,15])
plt.xlim([0,180])
plt.grid('both')
plt.ylabel('Energy [keV]')
plt.xlabel('theta °')
plt.legend()




plt.show()
