
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

    
    calP0_erik= 0.021307685642436364
    calP1_erik=0.020009508231844345

    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    
    
    common_path_erik='/home/maldera/Desktop/eXTP/meccanica-TO/McPherson-642/misure_mcPherson/run2/'
    common_pathTO='/home/maldera/Desktop/eXTP/datiSDD/mcPhersonNov2022/'

    #Fe
    #mca=['Fe_10kV_0ma.csv']#,'Fe_10kV_1ma.csv']
    #mca_TO=['Fe_10KV_0.0mA.mca','Fe_10KV_0.1mA.mca', 'Fe_10KV_0.0mA_orizzontale.mca' ]  #,'Fe_8KV_0mA.mca']
    #linesLabels=['Fe Ka','Fe kb']
    #norm_limits=[6.2,6.6]  # Fe ka
    #title='normalized @ Fe K alpha'
    
    #Ti
    #linesLabels=['Ti ka','Ti kb']
    #mca=['Ti_7kv_0ma.csv', 'Ti_7kv_1ma.csv']
    #mca_TO=['Ti_7KV_0.0mA.mca','Ti_10KV_0.0mA_orizzontale.mca']
    #norm_limits=[4.3,4.7]  # Ti ka
    #title='normalized @ Ti K alpha'
    
    #Ni
    #linesLabels=['Ni ka','Ni kb']
    #norm_limits=[7.3,7.6]  # Ni ka
    #mca=['Ni_10kV_0ma.csv',      'Ni_10kV_1ma.csv', 'Ni_10kV_1ma_air.csv']# ,'Ni_10kV_5ma_air.csv']
    #mca_TO=['Ni_10KV_0.0mA.mca','Ni_10KV_0.0mA_orizzontale.mca']
    #title='normalized @ Ni K alpha'

    #Rh    
    linesLabels=['Rh La']
    norm_limits=[2.64,2.74]  # Rh La
    mca=['Rh_10kV_0ma.csv']
    mca_TO=['Rh_10KV_0.0mA.mca','Rh_10KV_0.0mA_orizzontale.mca']
    title='normalized @ Rh L alpha'
     
    #linesLabels=['Fe Ka','Fe kb','Ti ka', 'Ti kb','Ni ka', 'Ni kb', 'Mo La', 'Rh La','Pd La','Pd Lb']   
    
    
           

        
        
        
    fig=plt.figure(1,figsize=(15,10))
    # plot McPhearson data 
    for i in range (0,len(mca)):
        
         p=histogramSimo()
         filename=common_path_erik+mca[i]            
         p.read_from_file(filename, 'Eric_mcPherson' )
       
         p.bins=p.bins*calP1_erik+calP0_erik
         p.normalize(norm_limits[0],norm_limits[1])
         mylabel=mca[i][0:-4]      
         plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts, histtype='step', label=mylabel+'@McPherson')

    # now plot SDD Torino data      
    for i in range (0,len(mca_TO)):
        
         p=histogramSimo()
         filename=common_pathTO+mca_TO[i]            
         p.read_from_file(filename, 'sdd' )
        
         p.bins=p.bins*calP1+calP0
         
         p.normalize(norm_limits[0],norm_limits[1])
         mylabel=mca_TO[i][0:-4]
         plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts, histtype='step', label=mylabel+'@TO',alpha=0.8)
       


         
       # col = mpl.cm.jet([0.25,0.75])  
    
    n = len(linesLabels)
    colors = mpl.cm.jet(np.linspace(0,1,n))

    for kk in range (0,len(linesLabels)):
           label=linesLabels[kk]
           plt.axvline( x=linesDict[label],label=label,linestyle='--',color=colors[kk]  )  

    #plot
    plt.xlabel('keV')
    plt.ylabel('norm. counts]')
    plt.legend()  
    plt.title(title)
    
    plt.show()


       



