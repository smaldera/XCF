from AmptekControl import SDD
from matplotlib import pyplot as plt
import numpy as np
import argparse as ar

formatter = ar.ArgumentDefaultsHelpFormatter
parser = ar.ArgumentParser(formatter_class=formatter)
parser.add_argument('-path', type=str, help='npz file path', default='./', required=False)
parser.add_argument('-time', type=str, help='livetime', default='10', required=False)
parser.add_argument('-name', type=str, help='name', default='', required=False)
args = parser.parse_args()
if args.path == './':
    path = '/mnt/c/Users/XCF/'
else:
    path = '/mnt/c/Users/XCF/' + args.path
print(path)

time = float(args.time)
print(time)
print(type(time))
Amp = SDD()
data, utilData = Amp.SaveAndAcquire(livetime=time, path=path, name=args.name)
