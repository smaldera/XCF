import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os.path as ospath
from astropy.time import Time

import sys
sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
import matplotlib as mpl

mpl.rcParams['font.size']=15  #!!!!!!!!!!!!!!!!!!!!!!!!!!

   

def plot_all(base_path,input_files,legend,norm_limits,x_lines,labels,labels_scale,calP1=1.5005121319323431 ,calP0=-19.137179036724653):
    

    fig=plt.figure(1, (13,10),tight_layout=True)
    ax = fig.subplots()

    for i in range(0,len(input_files)):
        filename=ospath.join(base_path,input_files[i])
        p=histogramSimo()
        p.read_from_file(filename,'sdd'  )
        
        p.bins=p.bins*calP1+calP0
        p.normalize(norm_limits[i][0],norm_limits[i][1])
        p.bins=p.bins/1000.
        p.plot(ax,legend[i])

        for j in range(0, len(x_lines[i])):

            hor_al='right'
            if legend[i]=='Si111' and j==0 :
                hor_al='left'
                print("LEFT!")
            plt.vlines(x=x_lines[i][j]/1000., ymin=0.001,ymax=labels_scale[i][j]*1.2,color='lightgray', linestyle='--')
            ax.text(x_lines[i][j]/1000.,labels_scale[i][j]*1.24 ,labels[i][j], horizontalalignment=hor_al, verticalalignment='center', size=10, color='gray')
        
        
        
    plt.xlabel('Energy [keV]')
    plt.ylabel('Normalized counts')
    #plt.ylim(0,1.4)
   # plt.yscale('log')
    plt.ylim(0.001,1.55)
    plt.xlim(200/1000.,13000/1000.)
    
    plt.legend()

    fig.savefig("allxtals_spectra_lin.png")
    plt.show()
            

if __name__ == "__main__":

    BASE_PATH='sdd_data/'

    input_files=['MXR_18kV_0.7mA_Si400.mca','MXR_20kV_0.7mA_Ge422.mca','MXR_15kV_0.6mA_Ge111.mca','MXR_15kV_0.6mA_Si220.mca','MXR_15kV_0.6mA_Si111.mca']
    legend=['Si400','Ge422','Ge111','Si220','Si111']
    norm_limtis=[[6250,6425],[7350,7525],[2550, 2675],[4400, 4550],[2780,2880]]
    x_lines=[[6332,1740],[7443,9886,10982],[2631,2631*3,2631*4],[4450,4450*2],[2835,2835*3,2835*4]]
    labels=[["Si400\nn=1",r'Si K$\alpha$'],['Ge442\nn=1',r'Ge K$\alpha$',r'Ge K$\beta$'],['Ge111\nn=1','Ge111\nn=3','Ge111\nn=4'],['Si200\nn=1','Si200\nn=2'],['Si111\nn=1 ','Si111\nn=3','Si111\nn=4']]
    labels_scale=[[1,0.4],[1,0.4,0.4],[1.1,1.1,1.1],[1,1],[0.9,0.9,0.9]]
   
    plot_all(BASE_PATH,input_files,legend,norm_limtis,x_lines,labels,labels_scale)
    

    

