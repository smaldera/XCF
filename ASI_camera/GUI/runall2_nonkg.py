import os.path
import os

import argparse
import zwoasi as asi


#base_path='/home/xcf/testCMOS_verticale/new7/'
base_path='/data/testPolSpot/testLong/'
bkg_path='/data/testPolSpot/bkg/'

sample_size=10000
bkg_sample_size=300
EXPO=100000 # us
camera_id=0

### creare loop principale
for i in range(0,100):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)+'/'

    cmd='python runOne_noBkg.py -o '+StoreDataIn+' -s '+str(sample_size)+' -bkg_path '+bkg_path+ ' -exp '+str(EXPO) +' -id '+str(camera_id)
    os.system(cmd)
        

print("... all done")


        
