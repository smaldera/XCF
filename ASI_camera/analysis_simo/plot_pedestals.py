from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../libs')

import utils as al
import ROOT
rootObjects=[]




# sensore 2 G120 1s
file_mean=['/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/sensorPXR/G120_10ms_bg/mean_ped.fits', '/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/G120_10ms_bg/mean_ped.fits']
file_std=['/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/sensorPXR/G120_10ms_bg/std_ped.fits', '/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/G120_10ms_bg/std_ped.fits']
leg_names=['PXR','eureka']



c1=ROOT.TCanvas('c1','',0)         
c1.Divide(2)

h_mean=[ROOT.TH1F()]*len(file_mean)
h_std=[ROOT.TH1F()]*len(file_mean)

leg=ROOT.TLegend(0.2,0.6,0.7,0.8)
legRMS=ROOT.TLegend(0.2,0.6,0.7,0.8)

for i in range (0, len(file_mean)):
    mean= al.read_image(file_mean[i])
    std= al.read_image(file_std[i])

    #myMask=np.where(std<7.5)
    myMask=np.where(std<700000000000)
   
    #myMask=np.where( mean>500)
 
   # al.plot_image(mean)
    #al.isto_all(mean)
    #al.plot_image(std)
    #al.isto_all(std)

   # plt.plot(std.flatten(),mean.flatten(),'bo',alpha=0.1)
   # plt.show()

    #print ('len mean[mask]= ',len(mean[myMask])) 
   
    h_mean[i]=al.isto_all_root(mean[myMask])
    h_mean[i].SetName('meanPed_'+str(i))
    h_mean[i].SetTitle('mean pedestal')
    h_mean[i].GetXaxis().SetTitle('ADC ch.')
    h_mean[i].SetLineColor(i+2)
    leg.AddEntry(h_mean[i],'mean '+leg_names[i],'l')
    
    h_std[i]=al.isto_all_root(std[myMask])
    h_std[i].SetName('stdPed_'+str(i) )
    h_std[i].SetTitle('pedestal RMS')
    h_std[i].GetXaxis().SetTitle('ADC ch.')
    h_std[i].SetLineColor(i+2)
    legRMS.AddEntry(h_std[i],'std dev. '+leg_names[i],'l')
   
    if (i==0):
         c1.cd(1)
         h_mean[i].Draw('hist')
         c1.cd(2)
         h_std[i].Draw('hist')
    else:
         c1.cd(1)
         h_mean[i].Draw('sames')
         c1.cd(2)
         h_std[i].Draw('sames')

        
        
        

c1.cd(1)
leg.Draw()
c1.cd(2)
legRMS.Draw()


c1.Update()
#c1.Draw()


#plt.imshow(mean[myMask] )
#print (myMask)
#plt.plot(myMask[0],myMask[1],'ro')
#plt.colorbar()
#plt.show()

# wait for stop:   
#input('press any key to continue...')
     
