#import os.path
import os
#import subprocess
import argparse
import zwoasi as asi
#import configparser
#from cmos_pedestal import bg_map
from cmos_pedestal import bg_map_rt
#from cmos_pedestal import capture
#from utils_v2 import read_image

from gui_analyzer_parallelSaveLargeClu import aotr2


formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-o','--out_path', type=str,  help='out path', required=True)
parser.add_argument('-s','--sample_size', type=int,  help='n. images', required=True)
parser.add_argument('-b','--bkg_size', type=int,  help='n. images background', required=True)
parser.add_argument('-exp','--exposure', type=int,  help='expusure time (us)', required=True)
parser.add_argument('-id','--camera_id', type=int,  help='camera_id', required=True)


args = parser.parse_args()



base_path=args.out_path
sample_size=args.sample_size
bkg_sample_size=args.bkg_size

WB_R=50
WB_B=50
#EXPO=300000 # us
EXPO=args.exposure
GAIN=120
CAMERA_ID=args.camera_id

print ("out path= ",base_path)
print ("sample_size= ",sample_size)
print ("bkg size= ",bkg_sample_size)
print ("Exposure= ",EXPO)
print ("camera_id= ",CAMERA_ID)




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


StoreDataIn=base_path
bkg_folder_a=StoreDataIn+'/bkg/'
cmd='mkdir -p '+StoreDataIn
os.system(cmd)

cmd='mkdir -p '+bkg_folder_a
os.system(cmd)

    
# create pedestal:
print("getting bkg")
path_to_bkg=bkg_folder_a


try: 
   bg_map_rt(path_to_bkg, path_to_bkg + '/mean_ped.fits', path_to_bkg + '/std_ped.fits', bkg_sample_size,  GAIN, WB_B, WB_R, EXPO,camera_id=CAMERA_ID, GUI=False)
   print("bkg created in: ",path_to_bkg)

    
   print("inizialising acq loop:")
   OBJ = aotr2(StoreDataIn, sample_size, WB_R, WB_B, EXPO, GAIN, bkg_folder_a, xyRebin, sigma, cluster, ApplyClustering, SaveEventList,raw, Eps,num_jobs ,leng,save_every,camera_id=CAMERA_ID, LIVE_PLOTS=False,GUI=False)
   print("starting acq loop:")  
   OBJ.CaptureAnalyze() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

except Exception as e:
    print("\n There are troubles in CaptureAndAnalyze: ", e)
    

finally:
    print ("del OBJ")
    del OBJ
        
        

print("... all done")


        
