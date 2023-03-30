import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo


mpl.rcParams["font.size"] = 15


linesDict={'Fe Ka':6.4, 'Fe kb':7.058, 'Ti ka':4.51, 'Ti kb':4.93,'Ni ka':7.47, 'Ni kb':8.264, 'Mo La':2.29, 'Rh La':2.696,'Pd La':2.838, 'Pd Lb':2.990}



if __name__ == "__main__":

    
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    #common_path='/home/xcf/Desktop/datiSDD/misureMcPhersonGenn2023/'
    common_path='/home/maldera/Desktop/eXTP/data/datiSDD/misureMcPhersonGenn2023/'
  
    #Fe
    mca_file=['Fe_10KV_0.0mA_verticale.mca','Fe_10KV_0.1mA_orizzontale.mca']
    linesLabels=['Fe Ka','Fe kb']

    
    for i in range (0,len(mca_file)):

       p=histogramSimo()
       filename=common_path+mca_file[i]            
       p.read_from_file(filename, 'sdd' )
       print("livetime=",p.sdd_liveTime,"counts=", p.sdd_fastCounts, "RATE=",p.sdd_fastCounts/p.sdd_liveTime,' Hz' )
       print("deadTime=",p.sdd_deadTime)
       
       # calibrazione energia
       p.bins=p.bins*calP1+calP0
       
       mylabel=mca_file[i][0:-4]
       plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts/p.sdd_liveTime, histtype='step', label=mylabel)

    
       
   # col = mpl.cm.jet([0.25,0.75])  
    n = len(linesLabels)
    colors = mpl.cm.jet(np.linspace(0,1,n))

    for jj in range (0,len(linesLabels)):
       label=linesLabels[jj]
       plt.axvline( x=linesDict[label],label=label,linestyle='--',color=colors[jj]  )  

    #plot
    plt.xlabel('keV')
    plt.ylabel('counts/s [Hz]')
    plt.legend()  
    plt.title('SDD')
    plt.show()




