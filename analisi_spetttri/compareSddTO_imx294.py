
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

    calP0_imx=-0.0013498026638486778  #calP0Err= 3.3894706711692284e-05
    calP1_imx= 0.0032116875215051385   #calP1Err= 3.284553141476064e-08



    
    
    common_pathSDD='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/SDD/'
    common_pathImx='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/10mm/1s_G120/'
   

   
    
    #Ni
   # linesLabels=['Ni ka','Ni kb']
   # imx=['/mcPherson_orizz/Ni/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz']
   # imx_labels=['10kV_0mA_orizzontale_imx']
   # mca=['Ni_10KV_0.0mA_orizzontale_nuovafinestra.mca']
   # norm_limits=[7.3,7.6]  # Ti ka
   # title='normalized @ Ni K alpha'
    
   
    #Pd    
    #linesLabels=['Pd La']
    #norm_limits=[2.74,2.9]  # Rh La
    #imx=['mcPherson_orizz/Pd/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz'] #,  'mcPherson_verticale/Pd/1ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz']
    #imx_labels=['10kV_0mA_orizzontale_imx','10kV_0mA_verticale_imx']
    #time_imx=[100*100*1e-3,100*1e-3]
    #mca=['Pd_10KV_0.0mA_orizzontale.mca']
    #mca=[]
    #title='normalized @ Pd L alpha'
    #title='events/s'
    #linesLabels=['Fe Ka','Fe kb','Ti ka', 'Ti kb','Ni ka', 'Ni kb', 'Mo La', 'Rh La','Pd La','Pd Lb']   

    #sorgente Fe
    imx=['spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz']
   
    time_imx=[100.*1]
    imx_labels=['Fe IMX294']
    mca=['Fe_14Oct2022_5mm.mca']
    linesLabels=[]
    title='events/s'
    norm_limits=[5.8,6]       
    title='normalized @ 55Fe Kalpha'
        
        
        
    fig=plt.figure(1,figsize=(15,10))
    # ploSDD
    for i in range (0,len(mca)):
        
         p=histogramSimo()
         filename=common_pathSDD+mca[i]            
         p.read_from_file(filename, 'sdd' )
       
       
         p.bins=p.bins*calP1+calP0
         p.normalize(norm_limits[0],norm_limits[1])
       #  p.couts=p.counts/p.sdd_liveTime
         
         mylabel=mca[i][0:-4]      
         plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts, histtype='step', label=mylabel+'_sdd',alpha=0.9)

    # now plot imx     
    for i in range (0,len(imx)):
        
         p=histogramSimo()
         filename=common_pathImx+imx[i]            
         p.read_from_file(filename, 'npz' )
        
         p.bins=p.bins*calP1_imx+calP0_imx
        # p.couts=p.counts/time_imx[i]
         p.normalize(norm_limits[0],norm_limits[1])
         mylabel=imx[i][0:-4]
         plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts, histtype='step', label=imx_labels[i],alpha=0.8)
       


         
       # col = mpl.cm.jet([0.25,0.75])  
    
    n = len(linesLabels)
    colors = mpl.cm.jet(np.linspace(0,1,n))

    for kk in range (0,len(linesLabels)):
           label=linesLabels[kk]
           plt.axvline( x=linesDict[label],label=label,linestyle='--',color=colors[kk], alpha=0.4  )  

    #plot
    plt.xlabel('keV')
    plt.ylabel('counts/s')
    plt.legend()  
    plt.title(title)
    
    plt.show()


       



