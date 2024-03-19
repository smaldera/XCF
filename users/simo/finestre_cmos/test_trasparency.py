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


    plt.figure(2)
    x=fitSimo.get_centers(pPRC.bins)
    y=(pPRC.counts/pAir.counts)

    y2=(pGPD.counts/pAir.counts)

    yRatio=pGPD.counts/pPRC.counts
    
    plt.plot(x,y,'ro',label='RPC/Air')
    plt.plot(x,y2,'bo',label='GPD/Air')
    plt.plot(x,yRatio,'ko',label='GPD/PRC')



    
    plt.legend() 
    

    plt.show()




