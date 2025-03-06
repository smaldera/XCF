import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo



calP0=-0.003201340833319255
calP1=0.003213272145961988
w_si=3.62

common_path='/home/maldera/Desktop/eXTP/data/CMOS_verticale/test_noise/1ms_G120/'
files_histo=['spectrum_all_raw_pixCut10.0sigma5_parallel.npz','spectrum_all_ZeroSupp_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.npz','spectrum_all_eps1.5_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.npz']
leg_names=['raw','soglia','clustering']

fig, ax = plt.subplots()
#for i in range(0,len(files_histo)):
for i in range(0,1):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')
    #p.counts=p.counts
    #p.bins=(p.bins*calP1+calP0)
    p.bins=1000.*(p.bins*calP1+calP0)/w_si

    initial_pars=[1e9,0,5]
   # popt1,pcov1,xmin1,xmax1, redChi1=fitSimo.fit_Gaushistogram(p.counts, p.bins,xmin=-10,xmax=10,initial_pars=initial_pars)
    popt1,pcov1, redChi1=fitSimo.fit_Gaushistogram(p.counts, p.bins,xmin=-7.5,xmax=6,initial_pars=initial_pars)
   
    print("popt1=",popt1)
    print("pcov1=",pcov1)
    print('recuced Chi2=',redChi1)
    #plot fitted function
    x=np.linspace(-7.5,6.,1000)
    y= fitSimo.gaussian_model(x,popt1[0],popt1[1],popt1[2])
    #y= fitSimo.gaussian_model(x, initial_pars[0],initial_pars [1],initial_pars[2])
   
    p.plot(ax,leg_names[i])
    plt.plot(x,y,'r-',label='fitted function')    
    
plt.title('test CMOS')
plt.xlabel('electrons')
ax.set_xlim(-100,100)

plt.ylabel('ounts')
plt.legend()
plt.show()

