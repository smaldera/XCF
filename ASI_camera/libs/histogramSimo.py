import numpy as np
from matplotlib import pyplot as plt
#sys.path.insert(0, '../../libs')
import fit_histogram as fitSimo


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

      def get_from_file(self,filename):
            data=np.load(filename)
            counts=data['counts']
            bins=data['bins']
            self.counts=counts
            self.bins=bins

      def normalize(self, minX,maxX):      
            bin_centers=fitSimo.get_centers(self.bins)
            mask=np.where((bin_centers>minX)&(bin_centers<maxX))
            c2=self.counts[mask]
            ka_h=np.max(c2)
            print("Ka max=",ka_h) 

            self.counts=self.counts/ka_h
            
      def plot(self,ax,labelname):

            histo=ax.hist(self.bins[:-1],bins=self.bins,weights=self.counts, histtype='step', label=labelname)


            
