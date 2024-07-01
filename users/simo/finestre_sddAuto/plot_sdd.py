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
from glob import glob
import datetime as dt

mpl.rcParams["font.size"] = 15


def compute_Ratios(c1,c2):
 
       s1=np.sqrt(c1)
       s2=np.sqrt(c2)
       r=c1/c2
       rErr=np.sqrt( (s1**2)*((1/c2)**2)+(s2**2)*(c1/(c2**2))**2)
       return r, rErr

def get_all_path(base_path,prefix):

       mypath=base_path+'/'+prefix+'*.npz'
       file_list=glob(mypath)
       print('file_list=',file_list)
       return file_list

def get_HistSpectrum(filesList):
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926
    pList=[]

    for filename in filesList:
        print("reading file->",filename)
        p=histogramSimo()
        p.read_from_file(filename, 'sddnpz' )

        p.bins=calP0+calP1*p.bins
        p.rebin(100)
        pList.append(p)

    return pList     

   

if __name__ == "__main__":

    #calP0=-0.03544731540487446
    #calP1=0.0015013787118821926

    base_path='/home/maldera/Desktop/eXTP/data/test_finestre/SaveNpz/11_06_24/'
   
    fileList=[] 
    fileList.append(get_all_path(base_path,'Air'))
    fileList.append(get_all_path(base_path,'GPD'))
    fileList.append( get_all_path(base_path,'PRC'))
   

    windows=['Air','GPD','PRC']
    spec_list=[]
    axList=[]
    airRateList=[]
    airTimeList=[]

    n_measures=float(len(fileList[0]))
    n_bins=0
    print ("n_measures=",n_measures)
    
    for jj in range(0,len(windows)): 
    
        spec_list.append(get_HistSpectrum(fileList[jj]))
        n_bins=len(spec_list[0][0].counts)

        print("n_bins=",n_bins)
    
    
        fig, ax=plt.subplots()
        axList.append(ax)
        airRate=[]
        airTime=[]
        for mySpec in spec_list[jj]:
            mySpec.plot(axList[jj],'aaa')
            airRate.append(mySpec.sdd_fastCounts/mySpec.sdd_liveTime)
            airTime.append(dt.datetime.fromtimestamp(mySpec.sdd_start))
            plt.ylabel("counts")
            plt.xlabel("E [keV]")
            plt.title(windows[jj])
            
        airRateList.append(airRate)
        airTimeList.append(airTime)
    

    fig2, ax2=plt.subplots()
    for jj in range(0,len(windows)):
        ax2.plot(airTimeList[jj],airRateList[jj],'p',label=windows[jj])
        plt.ylabel("sdd rate")
        plt.xlabel("time")
    plt.legend()



    # rapporti:
    print("n. spettrii=",len( spec_list[0]), " ",len( spec_list[1]),len( spec_list[2])  )

    gpdTrasp=np.array([])
    prcTrasp=np.array([])
    comparison=np.array([])
    gpdTraspErr=np.array([])
    prcTraspErr=np.array([])
    comparisonErr=np.array([])

    ratiosArray=np.array([])
    gpdRatiosArray=np.array([])
    prcRatiosArray=np.array([])
    
    for kk in range(0,len(spec_list[0])):

        gpdRatios,gpdErrs=  compute_Ratios( spec_list[1][kk].counts,  spec_list[0][kk].counts)
        prcRatios,prcErrs=  compute_Ratios( spec_list[2][kk].counts,  spec_list[0][kk].counts)
        comparisonRatios,comparisonErrs=  compute_Ratios( spec_list[1][kk].counts,  spec_list[2][kk].counts)

       # gpdTrasp=  gpdRatios
       # prcTrasp=prcRatios

        print("kk=",kk,  '-->> ',comparisonRatios)
       
        ratiosArray=np.append(comparisonRatios, ratiosArray)
        gpdRatiosArray=np.append(gpdRatios, gpdRatiosArray)
        prcRatiosArray=np.append(prcRatios, prcRatiosArray)
      
        if kk==0:           
              gpdTrasp=  gpdRatios
              prcTrasp=  prcRatios
              comparison= comparisonRatios

              gpdTraspErr=  gpdErrs**2
              prcTraspErr=  prcErrs**2
              comparisonErr= comparisonErrs**2
        else:
              gpdTrasp+= gpdRatios
              prcTrasp+=  prcRatios
              comparison+=   comparisonRatios
              gpdTraspErr+=  gpdErrs**2
              prcTraspErr+=  prcErrs**2
              comparisonErr+= comparisonErrs**2

              
    gpdTraspErr=np.sqrt(gpdTraspErr)/(n_measures)
    prcTraspErr=np.sqrt(prcTraspErr)/(n_measures)
    comparisonErr=np.sqrt(comparisonErr)/(n_measures)

    print("reshaped=",ratiosArray.reshape(int(n_measures),n_bins).T)
    ratiosArray=ratiosArray.reshape(int(n_measures),n_bins).T
    prcRatiosArray=prcRatiosArray.reshape(int(n_measures),n_bins).T
    gpRatiosArray=gpdRatiosArray.reshape(int(n_measures),n_bins).T
  
    
    fig3, ax3=plt.subplots()
    x=fitSimo.get_centers(spec_list[1][0].bins)               

    
    yRatio=[]
    yRatioErr=[]

    gpdRatio=[]
    gpdRatioErr=[]

    prcRatio=[]
    prcRatioErr=[]

    
    for i in range(0,n_bins):
           yRatio.append(ratiosArray[i].mean())
           yRatioErr.append(ratiosArray[i].std()/np.sqrt(n_measures))
           gpdRatio.append(gpdRatiosArray[i].mean())
           gpdRatioErr.append(gpdRatiosArray[i].std()/np.sqrt(n_measures))
          
           prcRatio.append(prcRatiosArray[i].mean())
           prcRatioErr.append(prcRatiosArray[i].std()/np.sqrt(n_measures))
           

    
 
    ax3.errorbar(x,yRatio,yerr=yRatioErr,fmt='pr',label='error from RMS')
    ax3.errorbar(x,comparison/n_measures,yerr=comparisonErr,fmt='p',label='stat. errors')
    plt.xlabel("E [keV]") 
    plt.title("GPD/PRC")
    plt.legend()
    plt.grid()
    
    fig4, ax4=plt.subplots()

    ax4.errorbar(x,gpdRatio,yerr=gpdRatioErr,fmt='pr',label='gpd/air errors from RMS')
    ax4.errorbar(x,prcRatio,yerr=prcRatioErr,fmt='pb',label='prc/air errors from RMS')
     
    ax4.plot(x, gpdTrasp/n_measures,'p',label='gpd/air')               
    ax4.plot(x, prcTrasp/n_measures,'p',label='prc/air')               
    plt.xlabel("E [keV]") 
    plt.title("Be trasparency")
    plt.legend()
                    
    plt.show()    
    
