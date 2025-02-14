import os.path
import os

import argparse
import zwoasi as asi


base_path='/home/xcf/testCMOS_verticale/new2/'
sample_size=100
bkg_sample_size=10

EXPO=300000 # us
### creare loop principale
for i in range(0,100):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)+'/'

    cmd='python runOne.py -o '+StoreDataIn+' -s '+str(sample_size)+' -b '+str(bkg_sample_size)+ ' -exp '+str(EXPO)
    os.system(cmd)
        

print("... all done")


        
