import numpy as np
from PiMove import PiMikro
from Point import Point2D

Pi = PiMikro()


p1=Point2D(X=12,Y=0)
p2=Point2D(X=12,Y=18)


while 1:    
    print("vado a p1")
    Pi.fastReach(p1)
    print("vado a p2")
    Pi.fastReach(p2)
