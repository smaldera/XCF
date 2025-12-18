import os.path
import os

import argparse
import zwoasi as asi


#base_path='/home/xcf/testCMOS_verticale/new7/'
base_path='/data/testCMOS_coincidenze/29Jul25/camera0/'


sample_size=5000
bkg_sample_size=100
camera_id=0

EXPO=250000 # us
### creare loop principale
for i in range(0,500):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)+'/'

    cmd='python runOne.py -o '+StoreDataIn+' -s '+str(sample_size)+' -b '+str(bkg_sample_size)+ ' -exp '+str(EXPO)+' -id '+str(camera_id)
    os.system(cmd)
        

print("... all done")


        
