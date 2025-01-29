
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



StoreDataIn='/home/xcf/testCMOS_verticale/data6/'
bkg_folder_a='/home/xcf/testCMOS_verticale/data5/bkg/'



sample_size=50
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

OBJ = aotr2(StoreDataIn, sample_size, WB_R, WB_B, EXPO, GAIN, bkg_folder_a, xyRebin, sigma, cluster, ApplyClustering, SaveEventList,raw, Eps,num_jobs ,leng,save_every,LIVE_PLOTS=False)

#try:
OBJ.CaptureAnalyze() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("Analysis has been completed and files are now stored in: " + StoreDataIn)
#except Exception as e:
#    print("\n There are troubles in CaptureAndAnalyze: ", e)
    


        
