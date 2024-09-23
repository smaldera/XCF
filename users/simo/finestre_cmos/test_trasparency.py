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

def compute_Ratios(c1,c2,s1,s2):
       r=c1/c2
       rErr=np.sqrt( (s1**2)*((1/c2)**2)+(s2**2)*(c1/(c2**2))**2)
       return r, rErr


def leggi_fitResults(nomefile):
    n_line=0
    f=open(nomefile)
    norms=[]
    normsErr=[]
    means=[]
    sigmas=[]
    for line in f:
        print(line[:-1].split())
        splitted=line[:-1].split()
        if n_line==0:
            print (splitted[0].split("=")[1])
            print (splitted[1].split("=")[1])
        else:
            print("AAA ",splitted[0].split("=")[1], " ",splitted[1].split("=")[1], " ",splitted[2].split("=")[1] )
            norms.append(float(splitted[0].split("=")[1]) )
            normsErr.append(float(splitted[1].split("=")[1]) )
            means.append(float(splitted[2].split("=")[1]))
            sigmas.append(float(splitted[3].split("=")[1]))
           
        n_line+=1    
    print("norms list= ",norms )                  
    return (np.array(norms),np.array(normsErr), np.array(means),np.array(sigmas))                  

                          
if __name__ == "__main__":

    
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    common_path='/home/maldera/Desktop/eXTP/data/test_finestre/cmos/'
    fileGPD='data_GPD/spectrumPos_all.npz'
    fileAir='data_100_air/spectrumPos_all.npz'
    filePRC='data_100_PRC/spectrumPos_all.npz'

    fileFitAir='data_100_air/corrCalib_all.txt'
    normsAir,normsErrAir  ,meansAir, sigmasAir=leggi_fitResults(common_path+fileFitAir)

    fileFitGPD='data_GPD/corrCalib_all.txt'
    normsGPD,normsErrGPD, meansGPD, sigmasGPD=leggi_fitResults(common_path+fileFitGPD)

    fileFitPRC='data_100_PRC/corrCalib_all.txt'
    normsPRC,normsErrPRC,meansPRC, sigmasPRC=leggi_fitResults(common_path+fileFitPRC)

    ELa=2.2932
    ELb=2.3948
    ESi=1.74
    ELa_escape=ELa-ESi
    ELb_escape=ELb-ESi
    energies=np.array([ELa, ELb,ESi,ELa_escape,  ELb_escape ])

    
    print("normsAir=",normsAir)                        
    
    Nrebin=100
    
    pGPD=histogramSimo()
    pGPD.read_from_file(common_path+fileGPD, 'npz' )
    pGPD.rebin(Nrebin)


    pPRC=histogramSimo()
    pPRC.read_from_file(common_path+filePRC, 'npz' )
    pPRC.rebin(Nrebin)

    pAir=histogramSimo()
    pAir.read_from_file(common_path+fileAir, 'npz' )
    pAir.rebin(Nrebin)
   
    
       
    
    plt.hist(pPRC.bins[:-1],bins=pPRC.bins ,weights=pPRC.counts, histtype='step', label='Be PRC')
    plt.hist(pAir.bins[:-1],bins=pAir.bins ,weights=pAir.counts, histtype='step', label='Air')
    plt.hist(pGPD.bins[:-1],bins=pGPD.bins ,weights=pGPD.counts, histtype='step', label='Be GPD')


    #plot
    plt.xlabel('keV')
    plt.ylabel('counts')
    plt.legend()  
   # plt.title('SDD')

 
    
  
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
    

    yFit,  yFitErr= compute_Ratios(normsGPD,normsAir,normsErrGPD,normsErrAir)
    yFit2,  yFit2Err= compute_Ratios(normsPRC,normsAir,normsErrPRC,normsErrAir)
    
    ax.errorbar(x,y,yerr=yerr, fmt='ro',label='RPC/Air')
    ax.errorbar(x,y2,yerr=y2err,fmt='bo',label='GPD/Air')
    ax.errorbar(energies,yFit,yerr=yFitErr, fmt='ks',label="peak  ratios GPD/AIR")
    ax.errorbar(energies,yFit2,yerr=yFit2Err, fmt='gs',label="peak  ratios PRC/AIR")
   
    plt.grid()
    plt.legend() 



    fig3, ax3 = plt.subplots()
    fitRatio,fitRatioErr= compute_Ratios(normsGPD,normsPRC,normsErrGPD,normsErrPRC)
    ax3.errorbar(x,yRatio,yerr=yRatio_err,fmt='ko',label='GPD/PRC cmos ')
   # ax3.errorbar(energies,fitRatio,yerr=fitRatioErr, fmt='rs',label="peak  ratios GPD/PRC" )
    plt.grid()

    # get sdd data:
    nomefile='/home/maldera/Desktop/eXTP/data/test_finestre/t_Ratio_Windows_sdd.npz'
    data=np.load(nomefile)

    hRatioSdd=histogramSimo()
    hRatioSdd.bins=data['arr_0']
    hRatioSdd.counts=data['arr_1']
    
    hRatioSdd.plot(ax3,"ratio GPD/PRC  SDD_1 (Lorenzo) ")
    
    datasdAuto=np.load('/home/maldera/Desktop/eXTP/data/test_finestre/trapsrancySddAutoRms.npz')
    xSddAuto= datasdAuto['x']
    ySddAuto= datasdAuto['y']
    yErrSddAuto= datasdAuto['yerr']
    ax3.errorbar(xSddAuto,ySddAuto,yerr=yErrSddAuto,fmt='ro',label='GPD/PRC sddAuto')
    plt.grid()   
    plt.legend() 
    plt.xlim(1,6)
    plt.ylim(0.95,1.065)
    plt.grid()
    plt.show()




