import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from eventListAnalizer import readEventList







infile1='/home/maldera/IXPE/XCF/data/testAllimentoXtals/1/eventFileList.txt'
myReadEventList=  readEventList(infile1,path='/home/maldera/IXPE/XCF/data/testAllimentoXtals/1/' )
print("analizzo tutto!!!!!!")
myReadEventList.analyze_all(plot_all=False)

Nev=len(myReadEventList.w_all)
print("Nev = ",Nev)
