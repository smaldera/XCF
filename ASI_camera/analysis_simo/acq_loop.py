import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import subprocess
import time

from reduce_data2 import red_data
from pedestal import bg_map

exposure=10000 # in micro sec 
gain=117
n_shots=500
out_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/misureFe_13.7_imgN/'
asi_sw_path='/home/maldera/Desktop/eXTP/ASI294/ASI_linux_mac_SDK_V1.20.3/demo/test_simo/'

bkg_file='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/misureFe_11.7/bg_2/mean_ped.fits'

n_loop=0
while(1):
    n_loop+=1
    cmd=asi_sw_path+'pippo2 '+str(exposure)+' '+str(gain)+' '+str(n_shots)+' '+out_path
    print('going to run:',cmd)
    subprocess.call(cmd,shell=True)

    print ('ok!!!!!!!!!!!!!!!!!!!')
    # autobkg
    outMeanPed_file=out_path+'mean_ped.fits'
    outStdPed_file=out_path+'std_ped.fits'
    bg_map(out_path, outMeanPed_file, outStdPed_file, draw=0)
    bkg_file=  outMeanPed_file
    print ("auto bkg calcolato")
    
    out_name='reducedData_'+str(n_loop)
    red_data(out_path, bkg_file, out_name )

    if n_loop>300:
        break
