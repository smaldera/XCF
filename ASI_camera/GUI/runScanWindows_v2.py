import os
#import argparse
import zwoasi as asi
from cmos_pedestal import bg_map_rt
from gui_analyzer_parallelSaveLargeClu import aotr2
import time

#formatter = argparse.ArgumentDefaultsHelpFormatter
#parser = argparse.ArgumentParser(formatter_class=formatter)
#parser.add_argument('-o','--out_path', type=str,  help='out path', required=True)
#parser.add_argument('-s','--sample_size', type=int,  help='n. images', required=True)
#parser.add_argument('-b','--bkg_size', type=int,  help='n. images background', required=True)
#parser.add_argument('-exp','--exposure', type=int,  help='expusure time (us)', required=True)
#parser.add_argument('-id','--camera_id', type=int,  help='camera_id', required=True)
#parser.add_argument('-max_time','--max_time', type=int,  help='max_time', required=False)
#args = parser.parse_args()



base_path='/data/norcada/scan/'
sample_size=845
bkg_sample_size=200

WB_R=50
WB_B=50
EXPO=300000 # us
GAIN=120
CAMERA_ID=0
MAX_TIME=None #!!!!!!!!!!!!!!!!!!!!!!!!

print ("out path= ",base_path)
print ("sample_size= ",sample_size)
print ("bkg size= ",bkg_sample_size)
print ("Exposure= ",EXPO)
print ("camera_id= ",CAMERA_ID)
print ("max_time= ",MAX_TIME)




#xyRebin=5
#sigma=10
#cluster=10
#ApplyClustering=True
#SaveEventList=True
#raw=False
#Eps=1.5
#num_jobs=4   # num paraller jobs
#leng=500
#save_every=10
#env_filename = os.getenv('ZWO_ASI_LIB')
#asi.init(env_filename)


StoreDataIn=base_path
bkg_folder=base_path+'/bkg/'
cmd='mkdir -p '+StoreDataIn
os.system(cmd)

#cmd='mkdir -p '+bkg_folder
#os.system(cmd)
    
# create pedestal:
print("getting bkg")
path_to_bkg=bkg_folder

#bg_map_rt(path_to_bkg, path_to_bkg + '/mean_ped.fits', path_to_bkg + '/std_ped.fits', bkg_sample_size,  GAIN, WB_B, WB_R, EXPO,camera_id=CAMERA_ID, GUI=False)
print("bkg created in: ",path_to_bkg)
print("press c when ready to go: ",path_to_bkg)

breakpoint()
print("let's go!")


for n_meas in range (8,50 ):
   
   start=time.time()    
   print("======================>>>>>>>>> MISURA n. ",n_meas)

   print("press c when ready to measure AIR ")
   breakpoint()

   if  n_meas!=4:
      
      StoreDataIn=base_path+'air_'+str(n_meas)
      cmd='mkdir -p '+StoreDataIn
      os.system(cmd)
      
      cmd='python runOne_noBkg.py -o '+StoreDataIn+' -s '+str(sample_size)+' -bkg_path '+path_to_bkg +' -exp '+str(EXPO) +' -id '+str(CAMERA_ID)
      os.system(cmd)
    

      print("AIR done...  tempo acq=",time.time()-start)
      print("")
      

     
   print("press c when ready for NORCADA")
   breakpoint()
   
   start=time.time()  
   StoreDataIn=base_path+'win_'+str(n_meas)
   cmd='mkdir -p '+StoreDataIn
   os.system(cmd)


   cmd='python runOne_noBkg.py -o '+StoreDataIn+' -s '+str(sample_size)+' -bkg_path '+path_to_bkg +' -exp '+str(EXPO) +' -id '+str(CAMERA_ID)
   os.system(cmd)
      
   
   print("NORCADA done....  tempo acq=",time.time()-start)
   print("")
   



print("... all done")


        
