import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al

import fit_histogram as fitSimo

#import ROOT



#####################################################
common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/10mm/'
files_histo=['1s_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz','1s_G240/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz','1s_G280/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz']
leg_names=['G=120','G=240','G=280']


fig, ax = plt.subplots()

popt=[]
mean=[]
mean_err=[]
sigma=[]
sigma_err=[]


for i in range(0,len(files_histo)):
    
    data=np.load(common_path+files_histo[i])
    counts=data['counts']
    bins=data['bins']
    #print ("len(coutsAll)=",len(countsAll) )
    histo=ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step', label=leg_names[i])
 #   print ("bins= ",bins)
    bin_centers=fitSimo.get_centers(bins)
    
    # fit del picco principale:
    maskCut_fondo=np.where(bin_centers>1000)
    print("(counts[maskCut_fondo]=",counts[maskCut_fondo]  )
    peak=np.amax(counts[maskCut_fondo])
    print('peak=',peak)
    peak_bin=bin_centers[counts==peak][0]
    print('bin max=', peak_bin)
    
    initial_pars=[1000.,peak_bin,50.]
    popt1,pcov1,xmin1,xmax1=fitSimo.fit_Gaushistogram_iterative(counts, bins, peak_bin-100 ,peak_bin+100,initial_pars)

    mean.append(popt1[1])
    sigma.append(popt1[2])
    mean_err.append(pcov1[1][1]**0.5)
    print(pcov1)
    print('mean=',popt1[1], "sigma= ",popt1[2], " ris= ",popt1[2]/popt1[1])
   
    # plot fitted function
    x=np.linspace(int(xmin1), int(xmax1),1000)
    y= fitSimo.gaussian_model(x,popt1[0],popt1[1],popt1[2])
    plt.plot(x,y,'-',label='fitted function')
    
    
plt.title('exposure=1s, G=120, 100 frames, 5mm collimator')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()



######################3

G=np.array([10**(12./20),10**(24/20),10**(28/20)])
plt.figure(2)

yerr=[]
for i in range(0,3):
    yerr=(((1./mean[0])**2)*mean_err[i]**2+((mean[1]/(mean[0]**2))**2) *mean_err[0]**2)**0.5

plt.plot(G/G[0], np.array(mean)/mean[0],'or') 
plt.errorbar(G/G[0], np.array(mean)/mean[0] ,yerr=yerr, fmt='ro')
x=np.linspace(0,8,100)
y=x
plt.plot(x,y,'-b')
plt.xlabel('G nominale (10^(Gdb/20))/G0')
plt.ylabel('mean/mean0')
plt.show()

