import sys

import numpy as np
from matplotlib import pyplot as plt
import glob
sys.path.insert(0, '../../libs')
import utils_v2 as al
from cmos_pedestal import bg_map

if __name__ == "__main__":
   #bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2022-05-23_10_21_08Z'
   #bg_map(bg_shots_path,'mean_ped_no_n1.fits', 'std_ped_n1.fits', draw=1 )

  # bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/testFe/2022-02-11_11_56_05Z_src5sec'
  # bg_map(bg_shots_path,'mean_pedLong.fits', 'std_pedLong.fits', draw=0 )

  # bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1/noGlass/G0_32us/'
  # bg_map(bg_shots_path,bg_shots_path+'mean_pedLong.fits', bg_shots_path+'std_pedLong.fits', draw=1 )


  # bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/bkg/'
  # bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1 )


  # bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/CapObj/2022-06-20_13_06_01Z/'
  # bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1 )


  # bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/ASI_linux_mac_SDK_V1.20.3/demo/test_simo/'
  # bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1 )

 
 # bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/misureFe_11.7/bg_1/'
  #bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1 )

    
  #bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/misureFe_11.7/bg_2/'
  #bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1 )


  #bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/Fe/200us_0_50_50'
  #bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1 )

#  bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/200us_G0/'

#  bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/1s_G120_bg/'
  #bg_map(bg_shots_path,bg_shots_path+'mean_pedTEST.fits', bg_shots_path+'std_ped.fitsTest', draw=1, hist_pixel=[2819,1626] )
#  bg_map(bg_shots_path,bg_shots_path+'mean_pedTEST.fits', bg_shots_path+'std_ped.fitsTest', draw=1)


#   bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_verticale/1ms_G120_bg/'
#   bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1)

   bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_verticale/6_12/10ms_G120_bg_noFinestra/'
   bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1)
