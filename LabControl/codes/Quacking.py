import argparse
import numpy as np
import os
import pytz
from datetime import datetime
from GPDset   import MyDaq
from PiMove   import PiMikro
from Point    import Point2D

parser = argparse.ArgumentParser(description='Make GPD dance samba')
parser.add_argument('-acTime', type=int, default=3600, help='GPD time of acquisition')
parser.add_argument('-nRun', type=int, default=100, help='Number of acquisitions')
parser.add_argument('-width', type=str, default=1.2, help='Quacking width: how far can the position change from the centre')
parser.add_argument('-deg', type=int, default=0, help='The degree of rotation of UP axes in XCF coordinates')
parser.add_argument('-xlow', type=float, default=None, help='Sets xlow')
parser.add_argument('-ylow', type=float, default=None, help='Sets ylow')
parser.add_argument('-xup', type=float, default=None, help='Sets xup')
parser.add_argument('-yup', type=float, default=None, help='Sets yup')
parser.add_argument('-seed', type=int, default=None, help='Global random seed')
args = parser.parse_args()

def ControlOutput(path, starting_list):
    new_list = os.listdir(path)
    if new_list != starting_list:
        return True, new_list
    else:
        return False, starting_list

def DateFormat(Zone):
    date=datetime.now(pytz.timezone(Zone)).strftime("%Y-%m-%d %H %M")
    return date.replace(" ", "_").replace(".", "_").replace(":", "_").replace(",", "_")

Pi = PiMikro(seed = args.seed)
if args.xlow == None or args.ylow == None:
    Plow, _ = Pi.UnPolPosition(args.deg, move=False)
if args.xup == None or args.yup == None:
    _, Pup = Pi.UnPolPosition(args.deg, move=False)

Zone='Europe/Rome'
print(DateFormat(Zone))
LogName = "C:\DaqLogs\log_" + DateFormat(Zone) + ".txt"
IXPEpath='C:\XPEDATA'
Flag=True
listOfFiles = os.listdir(IXPEpath)
n0 = len(listOfFiles)
log_file = open(LogName, 'w')
Pi.LowReach(Plow)

while True:
    Pstart=Pup
    # normal width 1.2
    Pi.UniformQuake(width=args.width, Pup=Point2D(Pup.GetX(), Pup.GetY()))
    #Pi.UniformQuake(width=2, P0=Point2D(2.5, 19.5))
    
    if Flag:
        Flag=False
        #GPD.Acquire(time=args.acTime)
        StartPrint = f'Starting Acq {len(listOfFiles)-n0+1} - Start time {datetime.now(pytz.timezone(Zone)).time()}'
        print(StartPrint)
        log_file.write(StartPrint)
    Flag, listOfFiles = ControlOutput(IXPEpath, listOfFiles)
    if Flag:
        #GPD.PrintOut(log_file)
        log_file.write(f'File {listOfFiles[-1]} saved in {IXPEpath}\n')
    if len(listOfFiles)-n0 == args.nRun:
        break
