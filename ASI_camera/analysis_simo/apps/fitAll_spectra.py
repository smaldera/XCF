import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../../libs')
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
#import ROOT

def EB_theta(thetaB,dd):
    h=2.*np.pi*6.582119563e-16
    c=299792458.0*1e10
    E=h*c/(dd*np.sin(np.deg2rad(thetaB)))
    return E

#####################################################
common_path='/home/maldera/Desktop/eXTP/data/ASI_testMu/'
dirs=['9Febb2024_piomboH20/']
filename='test_spectrum.npz'
leg_names=['pb muons']
theta=[0]
fig, ax = plt.subplots()

popt=[]
mean=[]
mean_err=[]
sigma=[]
sigma_err=[]


for i in range(0,len(dirs)):
    
    #data=np.load(common_path+dirs[i]+filename)
    #counts=data['counts']
    #bins=data['bins']
    p=histogramSimo()
    p.read_from_file(common_path+dirs[i]+filename, 'npz' )
    p.rebin(6)

    counts=p.counts
    bins=p.bins
    
    print ("len(counts)=",len(counts) )
    histo=ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step', label=leg_names[i])
    print ("bins= ",bins)
    bin_centers=fitSimo.get_centers(bins)
    
    # fit del picco principale:
    #maskCut_fondo=np.where(bin_centers>10)
    #print("counts[maskCut_fondo]=",counts[maskCut_fondo]  )
    peak=np.amax(counts)
    print('peak=',peak)
    peak_bin=bin_centers[counts==peak][0]
    print('bin max=', peak_bin)
    
    initial_pars=[100.,peak_bin,0.2]
    popt1,pcov1,xmin1,xmax1, redChi2=fitSimo.fit_Gaushistogram_iterative(counts, bins, peak_bin-0.3 ,peak_bin+0.3,initial_pars)

    mean.append(popt1[1])
    sigma.append(popt1[2])
    mean_err.append(pcov1[1][1]**0.5)
    print(pcov1)
    print('mean=',popt1[1], "sigma= ",popt1[2], " ris= ",popt1[2]/popt1[1], "reduced chi2=",redChi2)

    print("xmin=",xmin1," xmax1",xmax1)
    print("p0=",popt1[0], "p1=",popt1[1],' p2=',popt1[2])
   
    
    # plot fitted function
    x=np.linspace(xmin1,xmax1,1000)
    y= fitSimo.gaussian_model(x,popt1[0],popt1[1],popt1[2])
       
    plt.plot(x,y,'-',label='fitted function')
   
    
plt.title('exposure=1s, G=120, 100 frames, 5mm collimator')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()



######################3


plt.figure(2)

plt.errorbar(90-np.array(theta)+1 , mean  ,yerr=mean_err, fmt='ro')

dd_Si220=3.840
x=np.linspace(40,50,10000)
y= EB_theta(x,dd_Si220)/1000.
plt.plot(x , y  ,'ko',label="bragg law")

plt.xlabel('theta presunto')
plt.ylabel('meanE')
plt.show()

