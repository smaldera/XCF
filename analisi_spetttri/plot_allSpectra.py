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




def  plotAllSpectra(InputFileName):

    f=open(InputFileName)
    p=histogramSimo()
    n_spectra=0
    base_path=''
    legend=''
    calP0=-0.0013498026638486778  #calP0Err= 3.3894706711692284e-05
    calP1= 0.0032116875215051385   #calP1Err= 3.284553141476064e-08
    time=1
    
    fig, ax = plt.subplots()
    for line in f:
        line=line.strip('\n')
       # print(line)
        if len(line)==0:
            continue
        if line[0]=='#':
            continue
        splitted=line.split('=')
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
            time=float(splitted[1])
       
            
            
        if splitted[0]=="ADD_PLOT":    
            # plot istogramma?
            p.bins=p.bins*calP1+calP0
            p.counts=p.counts/time
            p.plot(ax,legend)
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



