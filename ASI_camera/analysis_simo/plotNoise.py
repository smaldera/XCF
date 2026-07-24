import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
import textwrap



calP0=-0.003201340833319255
calP1=0.003213272145961988
w_si=3.62

#common_path='/home/maldera/IXPE/XCF/data/CMOS_verticale/test_noise/'
#files_histo=['1ms_G120/spectrum_all_raw_pixCut10.0sigma5_parallel.npz','300ms_G120/spectrum_all_raw_pixCut10.0sigma5_parallel.npz']



common_path='/home/maldera/IXPE/XCF/data/testReadNoise2/'
files_histo=['/1ms_G120/1/spectrum_all_raw_pixCut7sigma.npz', '/100ms_G120/1/spectrum_all_raw_pixCut7sigma.npz']
leg_names=['1 ms','100 ms']
loc=[2,6,3]
y_pos=[0.9,0.75]
myString=[]
fig=plt.figure(figsize=(10,10))
ax=plt.subplot(111)
#fig, ax = plt.subplots()
#for i in range(0,len(files_histo)):
ax.set_yscale('log')

for i in range(0,2):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')

    #p.counts=p.counts
    p.bins=p.bins*1000./w_si
    #p.bins=(p.bins*calP1+calP0)


  #  p.bins=1000.*(p.bins*calP1+calP0)/w_si

    initial_pars=[1e9,0,2]
    
    #popt1,pcov1, redChi1=fitSimo.fit_Gaushistogram(p.counts, p.bins,xmin=-10,xmax=10,initial_pars=initial_pars)
    popt1,pcov1,xmin,xmax,   redChi1=fitSimo.fit_Gaushistogram_iterative(p.counts, p.bins,xmin=-12,xmax=12,initial_pars=initial_pars,nSigma=4)
   
    print("popt1=",popt1)
 
    #plot fitted function  
    x=np.linspace(xmin,xmax,1000)
    y= fitSimo.gaussian_model(x,popt1[0],popt1[1],popt1[2])
       
    p.plot(ax,leg_names[i])
    plt.plot(x,y,'-')

    # show custom stat box
    mean,RMS=p.getMeanRms()
    mystring=leg_names[i]+' \n'+" "*50+" \n mean= "+str(round(mean,3))+"\n RMS="+str(round(RMS,3))+'\n'+" Gauss mean="+str(round(popt1[1],3))+r'$\pm$'+str(round(pcov1[1][1]**0.5,6))+"\n Gauss sigma="+str(round(popt1[2],3))+r'$\pm$'+str(round(pcov1[2][2]**0.5,6))

    

    # Wrap to a fixed character width
    #wrapped = textwrap.fill(mystring, width=20)
    wrapped =mystring
   
    
    bbox_props = dict(boxstyle='round', facecolor='white',  alpha=1)    
    ax.text(0.75, y_pos[i], wrapped ,  transform=ax.transAxes,   va='top', ha='left',  bbox=bbox_props) #,  bbox=dict(alpha=0.7))

    
plt.title('readout noise')
plt.xlabel('electrons')
ax.set_xlim(-100,100)

plt.ylabel('counts')
plt.legend(fontsize=12)
plt.show()

