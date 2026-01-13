import os.path
import os
import argparse
import zwoasi as asi

base_path='/data/cmos_Verticale_new/'

sample_size=10000
bkg_sample_size=500
EXPO=300000 # us
camera_id=0

### creare loop principale
for i in range(2000,3000):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)+'/'

    cmd='python runOne.py -o '+StoreDataIn+' -s '+str(sample_size)+' -b '+str(bkg_sample_size)+ ' -exp '+str(EXPO) +' -id '+str(camera_id)
    os.system(cmd)



print("... all done")


        
