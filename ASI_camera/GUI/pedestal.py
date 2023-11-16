import sys
sys.path.insert(0, '../../libs')
from cmos_pedestal import bg_map
import argparse

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-in','--inFile', type=str,  help='txt file with list of FITS files', required=True)
parser.add_argument('-path', type=str,  help='path to the dir for images', required=True)
parser.add_argument('-name', type=str,  help='name of saved images', required=True)
args = parser.parse_args()

if __name__ == "__main__":
   print("working")

   bg_shots_path=args.inFile
  # bg_shots_path='/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/sensorPXR/G120_10ms_bg/
   bg_map(bg_shots_path,bg_shots_path+'mean_ped.fits', bg_shots_path+'std_ped.fits', args.path, args.name)
