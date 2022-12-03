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

    
    calP0= 0.021307685642436364
    calP1=0.020009508231844345
    
    common_path='/home/maldera/Desktop/eXTP/meccanica-TO/McPherson-642/misure_mcPherson/run2/'
    linesLabels=['Fe Ka','Fe kb','Ti ka', 'Ti kb','Ni ka', 'Ni kb', 'Mo La', 'Rh La','Pd La','Pd Lb']   
    
    
    #mca_file=['Fe_10kV_0ma.csv',  'Ni_10kV_0ma.csv',      'Ni_10kV_1ma.csv',      'Rh_10kV_0ma.csv',  'Ti_7kv_0ma.csv',  'Ti_7kv_2ma.csv', 'Fe_10kV_1ma.csv',  'Ni_10kV_1ma_air.csv', 'Ni_10kV_5ma_air.csv',  'Ti_10kv.csv',      'Ti_7kv_1ma.csv']
    

    mca1=['Fe_10kV_0ma.csv',  'Fe_10kV_1ma.csv']
    mca2=['Ni_10kV_0ma.csv',      'Ni_10kV_1ma.csv', 'Ni_10kV_1ma_air.csv', 'Ni_10kV_5ma_air.csv']
    mca3=['Ti_7kv_0ma.csv',  'Ti_7kv_2ma.csv',  'Ti_10kv.csv',      'Ti_7kv_1ma.csv']
    mca4=['Rh_10kV_0ma.csv']
    
    lines1=['Fe Ka','Fe kb']
    lines2=['Ni ka', 'Ni kb']
    lines3=['Ti ka', 'Ti kb']
    lines4=['Rh La']
    
    mca_all=[mca1,mca2,mca3,mca4]
    lines_all=[lines1,lines2,lines3,lines4]  
           
    n_fig=0
    for jj in range (0,len(mca_all)):
       n_fig+=1
       fig=plt.figure(n_fig,figsize=(15,10))
       mca_file=mca_all[jj]
       for i in range (0,len(mca_file)):
         fig=plt.figure(n_fig,figsize=(15,10))
         p=histogramSimo()
         filename=common_path+mca_file[i]            
         p.read_from_file(filename, 'Eric_mcPherson' )
       
         p.bins=p.bins*calP1+calP0
       
         mylabel=mca_file[i][0:-4]
         plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts, histtype='step', label=mylabel)

    
       
         # correzione attenuazione aria
         #bin_centers=fitSimo.get_centers(p.bins)
         #att=air_attenuation.attenuation_vs_d(bin_centers,3)
         #plt.hist(p.bins[:-1],bins=p.bins,weights=p.counts/(att), histtype='step', label=mylabel+" air att. corrected ")

       # col = mpl.cm.jet([0.25,0.75])  
       linesLabels=lines_all[jj]    
       n = len(linesLabels)
       colors = mpl.cm.jet(np.linspace(0,1,n))

       for kk in range (0,len(linesLabels)):
           label=linesLabels[kk]
           plt.axvline( x=linesDict[label],label=label,linestyle='--',color=colors[kk]  )  

       #plot
       plt.xlabel('keV')
       plt.ylabel('counts/s [Hz]')
       plt.legend()  
       plt.title('spettri Erik')
       plt.show()


       



