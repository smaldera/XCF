
import os.path
import os
import subprocess
import argparse
import zwoasi as asi
import configparser
from cmos_pedestal import bg_map
from cmos_pedestal import bg_map_rt
from cmos_pedestal import capture
from utils_v2 import read_image
from utils_v2 import plot_image
from utils_v2 import isto_all
from gui_analyzer_parallelSaveLargeClu import aotr2
import multiprocessing
import glob



base_path='/home/xcf/testCMOS_verticale/new/'


StoreDataIn='/home/xcf/testCMOS_verticale/data6/'
bkg_folder_a='/home/xcf/testCMOS_verticale/data6/bkg/'



sample_size=40
bkg_sample_size=20

WB_R=50
WB_B=50
EXPO=300000
GAIN=120

xyRebin=5
sigma=10
cluster=10
ApplyClustering=True
SaveEventList=True
raw=False
Eps=1.5
num_jobs=4   # num paraller jobs
leng=500
save_every=10


env_filename = os.getenv('ZWO_ASI_LIB')
asi.init(env_filename)


### creare loop principale

for i in range(0,2):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)
    bkg_folder_a=StoreDataIn+'/bkg/'
    cmd='mkdir -p '+StoreDataIn
    os.system(cmd)

    cmd='mkdir -p '+bkg_folder_a
    os.system(cmd)

    
    # create pedestal:
    print("getting bkg")
    path_to_bkg=bkg_folder_a

    try: 
        bg_map_rt(path_to_bkg, path_to_bkg + '/mean_ped.fits', path_to_bkg + '/std_ped.fits', bkg_sample_size, GAIN, WB_B, WB_R, EXPO,GUI=False)
        print("bkg created in: ",path_to_bkg)

    
        print("inizialising acq loop:")
        OBJ = aotr2(StoreDataIn, sample_size, WB_R, WB_B, EXPO, GAIN, bkg_folder_a, xyRebin, sigma, cluster, ApplyClustering, SaveEventList,raw, Eps,num_jobs ,leng,save_every,LIVE_PLOTS=False,GUI=False)

    
        print("starting acq loop:")
   
        OBJ.CaptureAnalyze() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #    print("Analysis has been completed and files are now stored in: " + StoreDataIn)
    except Exception as e:
        print("\n There are troubles in CaptureAndAnalyze: ", e)
        continue

        

print("... all done")


        
