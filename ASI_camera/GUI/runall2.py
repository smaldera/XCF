import os.path
import os

import argparse
import zwoasi as asi


#base_path='/home/xcf/testCMOS_verticale/new7/'
base_path='/home/maldera/Desktop/eXTP/data/test2coinc/run2/sensro1/'

sample_size=2000
bkg_sample_size=300
EXPO=300000 # us
camera_id=1

### creare loop principale
for i in range(0,50):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)+'/'

    cmd='python runOne.py -o '+StoreDataIn+' -s '+str(sample_size)+' -b '+str(bkg_sample_size)+ ' -exp '+str(EXPO) +' -id '+str(camera_id)
    os.system(cmd)
        

print("... all done")


        
