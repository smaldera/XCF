import numpy as np
from PiMove import PiMikro
import time
import ConnectToWSL

Tempo = '300' #NOTE: livetime MUST be passed like a string value
path = 'Desktop/SaveNpz/10_06_24/' #NOTE: this path needs to be relative to 'C/Users/XCF/'. So if you's like to save in Desktop, type 'Desktop/'
loops = 10
Pi = PiMikro()
Pi.MoveThat('ylow', 0.0, velocity=3)
Pi.MoveThat('xup', 24., velocity=3)
Pi.MoveThat('yup', 7.2134790, velocity=3)

xA = [300., 216.5, 126.]
labels = ['Air', 'PRC', 'GPD']

print('Moving to home position: ', labels[0])
Pi.MoveThat('xlow', xA[0], velocity=3)

for j in range(1, loops+1):
    i = 0
    for pos in xA:
        print('Moving to ', labels[i])
        Pi.MoveThat('xlow', pos, velocity=3)
        time.sleep(2)
        ConnectToWSL.callps1(path=path, livetime=Tempo, name=labels[i]+'_10kV_0.006mA')
        print('Saving spectrum')
        i+=1
