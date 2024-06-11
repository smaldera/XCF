import numpy as np
import time
import sys
import amptek_hardware_interface as Amp
from datetime import datetime as dt
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class SDD(object):

    def __init__(self):
        self.itSelf = Amp.AmptekHardwareInterface()
        self.itSelf.connectUSB(-1)
        print('FOUND')
        self.CalP0 = -0.03544731
        self.CalP1 = 0.0015013787

    def PingPong(self):
        P = self.itSelf.Ping()
        print(f'Ping: {P}')
        return P

    def GetGain(self):
        return self.itSelf.GetTextConfiguration('GAIN')

    def GetTime(self):
        return time.time()

    def Plotter(self, ax, status):
        x = np.linspace(0,8191,8192)
        x = x*self.CalP1 + self.CalP0
        y = np.array(self.itSelf.GetSpectrum())
        ax.grid()
        ax.set_xlim(0, x[-1])
        fSize = 16
        ax.set_xlabel('Energy [keV]', fontsize = fSize)
        ax.set_ylabel('Counts [#]', fontsize = fSize)
        ax.step(x, y, where='pre', color = 'red')

    def plotGauss(self, ax, amplitude, mean, sigma, nsigma=1):
        x = np.linspace(mean-nsigma*sigma,mean+nsigma*sigma,1000)
        y = amplitude * np.exp(-(x-mean)**2/(2*sigma**2))
        ax.plot(x,y,color='blue',marker='')

    def Acquire(self, livetime=10):
        self.itSelf.ClearSpectrum()
        self.itSelf.SetPresetAccumulationTime(livetime)
        print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA: {livetime}')
        self.itSelf.Enable()
        start = self.GetTime()
        print("Acquisition started")
        utilData = list()
        fig, ax = plt.subplots(figsize=(15, 8))
        while True:
            time.sleep(1)
            status = self.itSelf.updateStatus(-1)
            utilData = [status.AccTime(), status.FastCount(), status.SlowCount(), status.DeadTime(), status.FastCount()/status.AccTime(), start]
            self.Plotter(ax, status)
            y = np.array(self.itSelf.GetSpectrum())
            try:
                par, cov, chi2 = self.FitAround(y)
                if chi2 <=100:
                    self.plotGauss(ax, amplitude=par[0], mean=par[1], sigma=par[2], nsigma=3)
                box = dict(boxstyle='round', fc='white', ec='blue', alpha=1)
                ax.text(9, 0.6*y.max(),f"Accumulation Time: {np.round(status.AccTime(),2)}s\nFast Counts: {status.FastCount()}\nSlow Counts: {status.SlowCount()}\nDead Time: {np.round(status.DeadTime(),2)}%\nRate: {np.round(status.FastCount()/status.AccTime(), 2)} Hz\nMean: {np.round(par[1],3)} +- {np.round(np.sqrt(cov[1][1]),3)} keV\nSigma: {np.round(par[2],3)} +- {np.round(np.sqrt(cov[2][2]),3)} keV\nChi2: {np.round(chi2,2)}", fontsize = 14 , bbox=box)
            except:
                # create a box containing the key info
                box = dict(boxstyle='round', fc='white', ec='blue', alpha=1)
                ax.text(9, 0.6*y.max(),f"Accumulation Time: {np.round(status.AccTime(),2)}s\nFast Counts: {status.FastCount()}\nSlow Counts: {status.SlowCount()}\nDead Time: {np.round(status.DeadTime(),2)}%\nRate: {np.round(status.FastCount()/status.AccTime(), 2)} Hz", fontsize = 14 , bbox=box)
            plt.draw()
            plt.pause(1)
            plt.cla()
            if not status.IsEnabled():
                plt.close()
                print("")
                break
        self.itSelf.Disable()
        stop = self.GetTime()
        print("Acquisition finished")
        #par, cov, chi2 = self.FitAround(self.itSelf.GetSpectrum())
        #self.Plotter(ax,status)
        utilData.append(stop)
        return y, np.array(utilData)

    def SaveAndAcquire(self, livetime, path='./', name=None):
        data, utilData = self.Acquire(livetime)
        if name is None:
            name = f'file_{dt.now().year}_{dt.now().month}_{dt.now().day}_{dt.now().hour}_{dt.now().minute}'
        else:
            name = f'{name}_{dt.now().year}_{dt.now().month}_{dt.now().day}_{dt.now().hour}_{dt.now().minute}'
        oFile = path+name
        print(f'Saving Data in: {oFile}')
        np.savez(oFile, spectrum = data, utils = utilData)
        return data, utilData

    def FitAround(self, data, amplitude=None, mean=None, sigma=None):
        if amplitude is None:
            amplitude=data.max()
        if mean is None:
            mean=(np.where(data==data.max())[0][0])*self.CalP1 + self.CalP0
        if sigma is None:
            sigma = 0.04
        min_x = mean - 0.05
        max_x = mean + 0.05
        x = np.linspace(0,8192,8193)
        bins = x*self.CalP1 + self.CalP0
        initial_guess = [amplitude, mean, sigma]
        par, cov, chi2 = fit_Gaushistogram(data, bins,xmin=min_x, xmax=max_x, initial_pars=initial_guess, parsBoundsLow=-np.inf, parsBoundsUp=np.inf)
        return par, cov, chi2

def gaussian_model(x,p0,p1,p2):
       amplitude=p0
       peak=p1
       sigma=p2
       y=  amplitude * np.exp(-0.5 * ((x - peak)**2. / sigma**2.))
       return y

def get_centers(bins):
    centers=np.empty(0)
    for i in range(0,len(bins)-1):
        c=bins[i]+0.5*(bins[i+1]-bins[i])
        centers=np.append(centers,c)
    return centers

def fit_Gaushistogram(counts,bins,xmin=-100000,xmax=100000, initial_pars=[1,1,1], parsBoundsLow=-np.inf, parsBoundsUp=np.inf ):
    bin_centers =get_centers(bins)
    _mask = (counts > 0)&(bin_centers>xmin)&(bin_centers<xmax)
    y_data=counts[_mask]
    x_data=bin_centers[_mask]
    sigma=np.sqrt(y_data)
    #print("y_data=",y_data)
    #print("len(y_data)=",len(y_data))
    #print("Bounds=",(parsBoundsLow, parsBoundsUp ))
    popt, pcov = curve_fit(gaussian_model, x_data, y_data,p0=initial_pars,absolute_sigma=True, sigma=sigma, bounds=(parsBoundsLow, parsBoundsUp ), maxfev=5000)
    y_fit= gaussian_model(x_data,popt[0],popt[1],popt[2])
    chisq = (  ((y_data - y_fit)**2)/y_fit).sum()
    ndof= len(y_data) - len(popt)
    redChi2=chisq/ndof
    #print("fitting histo, chi2=",chisq," ndof=",ndof,"  red chi2=",redChi2)
    return popt,  pcov,redChi2
