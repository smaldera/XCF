import numpy as np
from PiMove import PiMikro

Pi = PiMikro()
Pi.UnPolPosition()
print('Ready? ')
input()
#Pi.MoveTheSnake(4, Pi.Mamba(10, 0.5), 1
#Pi.Cobra()
#Pi.HotSpot(0.675, -0.135)
Pi.Rattling(10, Pi.Mamba(7.5, 0.4), 0.7, maxShift=0.4)
