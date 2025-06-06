import os.path
import os

import argparse
import zwoasi as asi


base_path='/home/xcf/testCMOS_verticale/new7/'
sample_size=10000
bkg_sample_size=800

EXPO=300000 # us
### creare loop principale
for i in range(0,500):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)+'/'

    cmd='python runOne.py -o '+StoreDataIn+' -s '+str(sample_size)+' -b '+str(bkg_sample_size)+ ' -exp '+str(EXPO)
    os.system(cmd)
        

print("... all done")


        
