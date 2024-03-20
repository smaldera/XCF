import os
import argparse


# small script to run "read_eventLists.py"
# crea event list cercando il file npz nella cartella data
# lancia "read_eventLists.py" salvando nella cartella data

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('path', type=str,  help='data path')
args = parser.parse_args()

path=args.path
filename='events_list_pixCut10sigma_CLUcut_10sigma_v2.npz'
cmd='ls  '+path+filename+'  > '+path+'/file_list.txt'


print("sto per eseguire:",cmd)
os.system(cmd)

cmd='python read_eventLists.py -in '+path+"/file_list.txt  -dir "+path

print("sto per eseguire:",cmd)
os.system(cmd)

