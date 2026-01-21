import numpy as np
from matplotlib import pyplot as plt
#sys.path.insert(0, '../../libs')
import fit_histogram as fitSimo
import pandas as pd
from read_sdd import  pharse_mca

class histogramSimo():
      """
      primo tentativo di una classe per racchiudere tutte le funzioni che riguardano istogrammi 1D
      per ora utile solo per plottare e normalizzare
      possibili aggiunte (todo):
       - creazione con numpy passando l'array
       - salvataggio dei 2 vettori
       - calcolo media e rms
       - 
      """

      def __init__(self, counts=None, bins=None):

             self.counts=counts
             self.bins=bins
             self.sdd_liveTime=None
             self.sdd_deadTime=None
             self.sdd_fastCounts=None
             self.sdd_start=None

     # def get_from_file(self,filename):
     #       data=np.load(filename)
     #       counts=data['counts']
     #       bins=data['bins']
     #       self.counts=counts
     #       self.bins=bins

      def normalize(self, minX,maxX):      
            bin_centers=fitSimo.get_centers(self.bins)
            mask=np.where((bin_centers>minX)&(bin_centers<maxX))
            c2=self.counts[mask]
            ka_h=np.max(c2)
            print("Ka max=",ka_h) 

            self.counts=self.counts/ka_h
            
      def plot(self,ax,labelname):#,color='C0'):

            histo=ax.hist(self.bins[:-1],bins=self.bins,weights=self.counts, histtype='step', label=labelname)#, color=color)



      def read_from_file(self, filename, fileFormat ):

            data_array=None
     
            if fileFormat=='sdd':
                data_array, deadTime, livetime, fast_counts, start =pharse_mca(filename)
                size=len(data_array)      
                bin_edges=np.linspace(0,size+1,size+1)
                self.counts=data_array
                self.bins=bin_edges
                self.sdd_liveTime=livetime
                self.sdd_deadTime=deadTime
                self.sdd_fastCounts=fast_counts
                self.sdd_start=start
          
            if fileFormat=='Eric_mcPherson':
                 df=pd.read_csv(filename,header=None)
                 #x=df[0].values
                 y=df[1].values
                 size=len(y)      
                 bin_edges=np.linspace(0,size+1,size+1)
                 self.counts=y
                 self.bins=bin_edges


            if  fileFormat=='npz':
                 data=np.load(filename)
                 counts=data['counts']
                 bins=data['bins']
                 self.counts=counts
                 self.bins=bins

                 
     # def rebin(self,n):
            
