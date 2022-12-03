from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../libs')

import utils as al
import ROOT
rootObjects=[]



#file_mean=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1/noGlass/G0_32us/mean_pedLong.fits','/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1/noGlass/G0_32us_freddo/mean_pedLong.fits']
#file_std=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1/noGlass/G0_32us/std_pedLong.fits','/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1/noGlass/G0_32us_freddo/std_pedLong.fits']
#leg_names=['sensor1','sensor1 cold']

#file_mean=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/Fe55/bkg/mean_ped.fits']
#file_std=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/Fe55/bkg/std_ped.fits']
#leg_names=['sensor3']
#/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/Fe/200us_0_50_50/200us_0_50_50mean_ped.fits

#file_mean=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/Fe/200us_0_50_50/200us_0_50_50mean_ped.fits', '/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/200us_G0/mean_ped.fits']
#file_std=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/Fe/200us_0_50_50/200us_0_50_50std_ped.fits', '/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/200us_G0/std_ped.fits']
#leg_names=['sensor1_noGlass_FE','sensor1_noGlass_BKG']

# sensore 2 G120 32us
#file_mean=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_2/original/g120_32us/mean_ped.fits', '/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_2/noVetro/g120_32us_0offset/mean_ped.fits']
#file_std=['/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_2/original/g120_32us/std_ped.fits', '/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_2/noVetro/g120_32us_0offset/std_ped.fits']
#leg_names=['sensor2_original','sensor1_noGlass']

# sensore 2 G120 1s
file_mean=['/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_orizz/100ms_G120_bg/mean_ped.fits', '/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_verticale/100ms_G120_bg/mean_ped.fits']
file_std=['/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_orizz/100ms_G120_bg/std_ped.fits', '/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_verticale/100ms_G120_bg/std_ped.fits']
leg_names=['100ms orizz','100ms vert']



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
     
