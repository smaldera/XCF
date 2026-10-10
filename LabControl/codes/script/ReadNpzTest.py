import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

path = "C:/Users/XCF/Desktop/SaveNpztest1_2025_2_19_11_34.npz"

def gaussian_model(x,p0,p1,p2):
       amplitude=p0
       peak=p1
       sigma=p2
       y=  amplitude * np.exp(-0.5 * ((x - peak)**2. / sigma**2.))
       return y

def fit_Gaushistogram(x, y, xmin=-100000, xmax=100000, initial_pars=[1,1,1], parsBoundsLow=-np.inf, parsBoundsUp=np.inf ):
    _mask = (y>0)&(x>xmin)&(x<xmax)
    y_data=y[_mask]
    x_data=x[_mask]
    sigma=np.sqrt(y_data)
    popt, pcov = curve_fit(gaussian_model, x_data, y_data,p0=initial_pars,absolute_sigma=True, sigma=sigma, bounds=(parsBoundsLow, parsBoundsUp ), maxfev=5000)
    y_fit= gaussian_model(x_data,popt[0],popt[1],popt[2])
    chisq = (  ((y_data - y_fit)**2)/y_fit).sum()
    ndof= len(y_data) - len(popt)
    redChi2=chisq/ndof
    return popt,  pcov,redChi2

def FitAround(y, amplitude=None, mean=None, sigma=None):
    if amplitude is None:
        amplitude=y.max()
    if mean is None:
        mean=(np.where(y==y.max())[0][0])*m + q
    if sigma is None:
        sigma = 1
    min_x = mean - 0.05*sigma
    max_x = mean + 0.05*sigma
    x = np.linspace(0,8192,8192)
    print(len(x))
    print(len(y))
    x = x*m + q
    initial_guess = [amplitude, mean, sigma]
    par, cov, chi2 = fit_Gaushistogram(x, y, xmin=min_x, xmax=max_x, initial_pars=initial_guess, parsBoundsLow=-np.inf, parsBoundsUp=np.inf)
    return par, cov, chi2

data = np.load(path)
y = data['spectrum']
var = data['utils']

q = -0.03544731
m = 0.0015013787

"""
    utils contains the useful parameters of the aquisition:
    utilData = [ livetime [s], FastCount [#], SlowCount [#], DeadTime [%], Rate [Hz], start unixtime [s], stop unixtime [s] ]
"""
par, cov, chi = FitAround(y)
x = np.linspace(0,8192,8193)
x = m*x+q
fig = plt.figure(figsize=(16,10))
plt.plot(x[:-1],y)
#plt.ylim(bottom=0)
fs = 16
plt.grid(True, "both")
plt.title("SDD Spectrum", fontsize=fs+3, fontweight='bold')
plt.xlabel("Energy [keV]", fontsize=fs, fontweight='bold')
plt.ylabel("counts", fontsize=fs, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=fs-2)  # Cambia la dimensione dei tick (etichetta)
plt.text(0.75, 0.95,f'Livetime={var[0]} s\nRate={var[4]} Hz\nDeadTime={np.round(var[3],2)} %\nPeak={np.round(par[1],2)} keV\nSigma={np.round(par[2],2)} keV', transform=plt.gca().transAxes,
         horizontalalignment='left', verticalalignment='top', fontsize=fs, color='k', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
mask = np.where((par[1]-par[2]*2<x) & (par[1]+par[2]*2>x))
plt.plot(x[mask], gaussian_model(x[mask], par[0], par[1], par[2]))
plt.xlim(0, par[1]*3)
plt.show()
print(var)
print(y)
print(len(y))
