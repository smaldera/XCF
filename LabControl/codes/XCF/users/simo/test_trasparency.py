import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
#sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo


mpl.rcParams["font.size"] = 15




if __name__ == "__main__":

    
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    common_path='/home/maldera/Desktop/eXTP/data/datiSDD/MXR_Dic2023/Trasparenze/'
    fileGPD='1.200_HV_0.05_i_0.7mm_GPDwindow_10min_new.mca'
    filePRC='1.200_HV_0.05_i_0.7mm_PRCwindow_10min_new.mca'

    pGPD=histogramSimo()
    pGPD.read_from_file(common_path+fileGPD, 'sdd' )
    print("livetimeGPD=",pGPD.sdd_liveTime,"countsGPD=", pGPD.sdd_fastCounts, "RATE_GPD=",pGPD.sdd_fastCounts/pGPD.sdd_liveTime,' Hz' )
    print("deadTime=",pGPD.sdd_deadTime)


    pPRC=histogramSimo()
    pPRC.read_from_file(common_path+filePRC, 'sdd' )
    print("livetimePRC=",pPRC.sdd_liveTime,"countsPRC=", pPRC.sdd_fastCounts, "RATE_PRC=",pPRC.sdd_fastCounts/pPRC.sdd_liveTime,' Hz' )
    print("deadTime=",pPRC.sdd_deadTime)


    
    # calibrazione energia
    pGPD.bins=pGPD.bins*calP1+calP0
    pPRC.bins=pPRC.bins*calP1+calP0
   
    
    
    plt.hist(pPRC.bins[:-1],bins=pPRC.bins ,weights=pPRC.counts/pPRC.sdd_liveTime, histtype='step', label='Be PRC')
    plt.hist(pGPD.bins[:-1],bins=pGPD.bins ,weights=pGPD.counts/pGPD.sdd_liveTime, histtype='step', label='Be GPD')

    

    #plot
    plt.xlabel('keV')
    plt.ylabel('counts/s [Hz]')
    plt.legend()  
    plt.title('SDD')


    plt.figure(2)
    x=fitSimo.get_centers(pPRC.bins)
    y=(pGPD.counts/pGPD.sdd_liveTime)/(pPRC.counts/pPRC.sdd_liveTime)

    plt.plot(x,y,'ro',label='PDG/PRC') 
    plt.legend() 
    

    plt.show()




