import sys

import numpy as np
from matplotlib import pyplot as plt
import glob
sys.path.insert(0, '../../libs')
import utils_v2 as al
from cmos_pedestal import bg_map

if __name__ == "__main__":

   bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/test_bkg_novetro/'
  # bg_shots_path='/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/sensorPXR/G120_10ms_bg/
   bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', draw=1)
