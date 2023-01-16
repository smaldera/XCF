import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import subprocess
import time
import os.path

sys.path.insert(0, '../../libs')
#import utils as al
from reduce_data2 import red_data
from cmos_pedestal import bg_map




def run_acq_loop(exposure,gain, n_shots,nLoops,out_path,bkg_file):

    # mettere come configurazione!!!!! variabile d'ambiente???
    asi_sw_path='/home/maldera/Desktop/eXTP/ASI294/ASI_linux_mac_SDK_V1.20.3/demo/test_simo/'


    #checks:
    PedFile_exists = os.path.exists(bkg_file)
    if PedFile_exists==False:
        print('pedestal file:',bkg_file," not found!!! ...stop here!")
        exit()
    #create output folder:
    cmd="mkdir -p "+outFolder
    subprocess.call(cmd,shell=True)    


    
    n_loop=0
    while(n_loop<nLoops):
        n_loop+=1
        cmd=asi_sw_path+'pippo2 '+str(exposure)+' '+str(gain)+' '+str(n_shots)+' '+out_path
        print('going to run:',cmd)
        subprocess.call(cmd,shell=True)
        print ('ok!!!!!!!!!!!!!!!!!!!')

        # autobkg
        #outMeanPed_file=out_path+'mean_ped.fits'
        #outStdPed_file=out_path+'std_ped.fits'
        #bg_map(out_path, outMeanPed_file, outStdPed_file, draw=0)
        #bkg_file=  outMeanPed_file
            
        out_name='reducedData_'+str(n_loop)
        red_data(out_path, bkg_file, out_name )
        
        if n_loop>300:
            break



if __name__ == '__main__':
 
    import argparse
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument('-ex','--exposure', type=int, help='exp. time in microseconds',required=True)

    parser.add_argument('-n', '--num_events', type=int, default=100,  help = 'total n. of shots',required=True)
    parser.add_argument('-g', '--gain', type=int, default=120,  help = 'ASI camera gain',required=True)
    
    parser.add_argument('-nLoops', '--nLoops', type=int, default=30,  help = 'n. of loops',required=True)
    
    parser.add_argument('-p', '--pedestal',   help = 'pedestal file',required=True)
    parser.add_argument('-o', '--outfolder',  help = 'output folder',required=True)

    args = parser.parse_args()
    exposure=args.exposure
    nEvents=args.num_events
    pedFile=args.pedestal
    outFolder=args.outfolder
    gain=args.gain
    nLoops=args.nLoops
    
    print("going to run:")
    print("ped file: ",pedFile)
    print("outFolder: ",outFolder)
    print("gain: ",gain)
    print("exposure: ",exposure)
    print("n_events: ",nEvents)

      
    run_acq_loop(int(exposure),int(gain), int(nEvents),int(nLoops),outFolder,pedFile)
    
