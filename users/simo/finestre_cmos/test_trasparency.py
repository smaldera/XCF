import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '/home/maldera/Desktop/eXTP/softwareXCF/XCF/libs/')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo


mpl.rcParams["font.size"] = 15




if __name__ == "__main__":

    
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    common_path='/home/maldera/Desktop/eXTP/data/test_finestre/cmos/'
    fileGPD='data_GPD/test_spectrum.npz'
    fileAir='data_100_air/test_spectrum.npz'
    filePRC='data_100_PRC/test_spectrum.npz'

    pGPD=histogramSimo()
    pGPD.read_from_file(common_path+fileGPD, 'npz' )
   # print("livetimeGPD=",pGPD.sdd_liveTime,"countsGPD=", pGPD.sdd_fastCounts, "RATE_GPD=",pGPD.sdd_fastCounts/pGPD.sdd_liveTime,' Hz' )
   # print("deadTime=",pGPD.sdd_deadTime)


    pPRC=histogramSimo()
    pPRC.read_from_file(common_path+filePRC, 'npz' )
   # print("livetimePRC=",pPRC.sdd_liveTime,"countsPRC=", pPRC.sdd_fastCounts, "RATE_PRC=",pPRC.sdd_fastCounts/pPRC.sdd_liveTime,' Hz' )
   # print("deadTime=",pPRC.sdd_deadTime)


    pAir=histogramSimo()
    pAir.read_from_file(common_path+fileAir, 'npz' )
   # 
   
    
    # calibrazione energia
    #pGPD.bins=pGPD.bins*calP1+calP0
    #pPRC.bins=pPRC.bins*calP1+calP0
   
    
    
    plt.hist(pPRC.bins[:-1],bins=pPRC.bins ,weights=pPRC.counts, histtype='step', label='Be PRC')
    plt.hist(pAir.bins[:-1],bins=pAir.bins ,weights=pAir.counts, histtype='step', label='Air')
    plt.hist(pGPD.bins[:-1],bins=pGPD.bins ,weights=pGPD.counts, histtype='step', label='Be GPD')

    

    #plot
    plt.xlabel('keV')
    plt.ylabel('counts/s [Hz]')
    plt.legend()  
    plt.title('SDD')

##########################################333
# Rapporti

   
    
    fig2, ax = plt.subplots()
    x=fitSimo.get_centers(pPRC.bins)
    prc=pPRC.counts
    air=pAir.counts
    gpd=pGPD.counts

    s_prc=np.sqrt(prc)
    s_air=np.sqrt(air)
    s_gpd=np.sqrt(gpd)
        
    y=prc/air
    yerr=np.sqrt( (s_prc**2)*((1/air)**2)+(s_air**2)*(prc/(air**2))**2)
   
    
    y2=(pGPD.counts/pAir.counts)
    y2err=np.sqrt( (s_gpd**2)*((1/air)**2)+(s_air**2)*(gpd/(air**2))**2)
    
    yRatio=pGPD.counts/pPRC.counts
    yRatio_err=np.sqrt( (s_gpd**2)*((1/prc)**2)+(s_prc**2)*(gpd/(prc**2))**2)
    
    
    ax.errorbar(x,y,yerr=yerr, fmt='ro',label='RPC/Air')
    ax.errorbar(x,y2,yerr=y2err,fmt='bo',label='GPD/Air')
    plt.grid()
    plt.legend() 



    fig3, ax3 = plt.subplots()
    ax3.errorbar(x,yRatio,yerr=yRatio_err,fmt='ko',label='GPD/PRC')
    plt.grid()

    # get sdd data:
    nomefile='/home/maldera/Desktop/eXTP/data/test_finestre/t_Ratio_Windows_sdd.npz'
    data=np.load(nomefile)

    hRatioSdd=histogramSimo()
    hRatioSdd.bins=data['arr_0']
    hRatioSdd.counts=data['arr_1']
    
    hRatioSdd.plot(ax,"ratio SDD")
    
    
    
    
    plt.legend() 
    

    plt.show()




