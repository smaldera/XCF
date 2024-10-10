import argparse
import os
import pytz
from datetime import datetime
from GPDset   import MyDaq
from PiMove   import PiMikro
from Point    import Point2D

parser = argparse.ArgumentParser(description='Make GPD dance samba')
parser.add_argument('-acTime', type=int, default=60, help='GPD time of acquisition')
parser.add_argument('-nRun', type=int, default=1, help='Number of acquisitions')
parser.add_argument('-GPD_number', type=str, default=35, help='Which GPD are you using?')
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
_, P0 = Pi.UnPolPosition(deg=0)
#GPD = MyDaq(gpd = args.GPD_number)
Zone='Europe/Rome'
print(DateFormat(Zone))
LogName = "C:\DaqLogs\log_" + DateFormat(Zone) + ".txt"
IXPEpath='C:\XPEDATA'
Flag=True
listOfFiles = os.listdir(IXPEpath)
n0 = len(listOfFiles)
log_file = open(LogName, 'w')

while True:
    Pstart=P0
    Pi.UniformQuake(width=1., P0=Point2D(P0.GetX(), P0.GetY()))
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
