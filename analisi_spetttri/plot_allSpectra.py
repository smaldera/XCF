import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os.path as ospath

import sys
sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
import matplotlib as mpl

mpl.rcParams['font.size']=15  #!!!!!!!!!!!!!!!!!!!!!!!!!!




def  plotAllSpectra(InputFileName):

    f=open(InputFileName)
    p=histogramSimo()
    n_spectra=0
    base_path=''
    legend=''
    calP0=-0.0013498026638486778  #calP0Err= 3.3894706711692284e-05
    calP1=0.0032116875215051385   #calP1Err= 3.284553141476064e-08
   
    compute_rate=0 
    time=1
    normalize=0
    low=0.
    up=0.
    fig=plt.figure(1, (10,10))
    ax = fig.subplots()
    for line in f:

        
        line=line.strip('\n')
        line=" ".join(line.split()) # rimuove tutti gli spazi doppi????? 
       # print(line)
        if len(line)==0:
            continue
        if line[0]=='#':
            continue
        splitted=line.split('=')
        print("splitted[0]=",splitted[0])
        if splitted[0]=='BASE_PATH':
           base_path=splitted[1]
           print("BASE_PATH=",base_path)
        
        if splitted[0]=='FILE':

            file_split=splitted[1].split(' ')
            print('file_split',file_split)
            filename=ospath.join(base_path,file_split[0])
            fileFormat=file_split[1]
            print("reading file",filename)
            print ("file type = ",fileFormat)
            legend=filename.split('/')[-1]
            n_spectra+=1
            
            p.read_from_file(filename, fileFormat )

        if   splitted[0]=='LEGEND':   
              legend=splitted[1] 
              print ("legend=",legend)
        if splitted[0]=="P0":
            calP0=float(splitted[1])
        if splitted[0]=="P1":
            calP1=float(splitted[1])

                    
        if splitted[0]=="ACQ_TIME":
            compute_time=1
            if fileFormat=='sdd':
                time=p.sdd_liveTime
            else:    
                time=float(splitted[1])
       

        if  splitted[0]=="NORM_PEAK":
            #normalize peak height to 1
            print("!!!")
            normalize=1
            peak_limits=splitted[1].split(' ')
            if len(peak_limits)!=2:
                print("NORM_PEAK wrong limits... exit!")
                exit()
            low=float(peak_limits[0])
            up=float(peak_limits[1])
            if (up<=low):
                print("NORM_PEAK wrong limits... exit!")
                exit()
            print ('up=',up, "low = ",low)
            

                
 
        if splitted[0]=="ADD_PLOT":    
            # plot istogramma?
            p.bins=p.bins*calP1+calP0

            if compute_rate==1:
                 p.counts=p.counts/time
                 compute_rate=0
                 time=1
                 plt.ylabel('events/s')
            if normalize==1:
                 print("NORMALIZZO!!!!!")
                 p.normalize(low,up)
                 normalize=0
                 low=0.
                 up=0.
                 plt.ylabel('normalized rate')
                 
            p.plot(ax,legend)
            plt.xlabel('energy [keV]')
            
            plt.ylabel('events/s') # non so perche', ,ma nell'if non funziona!
            
            plt.legend()
          

      

   

       

            

if __name__ == "__main__":


    import argparse
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser( prog='calibrator.py',  formatter_class=formatter)
    parser.add_argument('infile', type=str, help='imput file')
    args = parser.parse_args()
    print('input file=',args.infile)
    #check file exist:
    if not (ospath.exists(args.infile)):
        print ("file not found:",args.infile)
        exit()
        
    plotAllSpectra(args.infile)
   

   
    plt.show()



