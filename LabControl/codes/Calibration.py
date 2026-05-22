import numpy as np
import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-D', type=str,  help='Detector', required=True, choices={'SDD','CMOS'})
parser.add_argument('-Chn', type=int,  help='Channel', required=True)

args = parser.parse_args()
Detector = args.D
Channel = args.Chn

def Cal(Detector, Channel):
  p0=0
  p1=0
  if Detector=='SDD':
    P0=-0.03544731
    p1=0.0015013787
  elif Detector=='CMOS':
    p0=-0.0032013
    p1=0.00321327
  else:
    print('Detector not in list')
    return(1)
  return Channel*p1+p0

print(f'{np.round(Cal(Detector, Channel),3)} keV')
