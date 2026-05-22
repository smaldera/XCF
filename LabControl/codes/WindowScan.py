import numpy as np
from PiMove import PiMikro
import time
import ConnectToWSL

Tempo = '300' #NOTE: livetime MUST be passed like a string value
Dt = 300+175
t_nor = 300 + 10
dt = 40
path = 'Desktop/SaveNpz/11_06_24/' #NOTE: this path needs to be relative to 'C/Users/XCF/'. So if you's like to save in Desktop, type 'Desktop/'
loops = 10
Pi = PiMikro()
Pi.MoveThat('ylow', 3., velocity=3)
Pi.MoveThat('xup', 16.0, velocity=3)
Pi.MoveThat('yup', 12.5, velocity=3)

xA = [300., 232.]
labels = ['Air', 'Norcada']

print('Moving to home position: ', labels[0], ' in ', xA[0] )
Pi.MoveThat('xlow', xA[0], velocity=3)
print('Ready\n--------------------------------------------------')
t = 0
for j in range(1, loops+1):
    i = 0
    for pos in xA:
        print('\nMoving to ', labels[i], ' in ', pos )
        move = time.time()
        Pi.MoveThat('xlow', pos, velocity=3)
        print('Arrived in position ', pos, ' in ', np.round(time.time()-move, 2), ' s')
        print('Acquisition started: ', labels[i])
        ac_start = time.time()
        print('\n!-!-------Press c if ready to move-------!-!')
        breakpoint()
        print('Acquisition + Analysis time: ', np.round(time.time()-ac_start, 2),'\n')
        i+=1
        #ConnectToWSL.callps1(path=path, livetime=Tempo, name=labels[i]+'_10kV_0.006mA')
        
