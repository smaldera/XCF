import argparse
import os
import sys
sys.path.insert(0,r"C:\Users\XCF\Desktop\codes")

from PiMove import PiMikro

def polPosCmos(Pi: PiMikro):
    print("Moving xLow")
    Pi.MoveThat("xlow", 148., velocity = 3)
    print("Moving yLow")
    Pi.MoveThat("ylow", 11.1, velocity = 3)
    print("Moving xUp")
    Pi.MoveThat("xup",  12.5, velocity = 3)
    print("Moving yUp")
    Pi.MoveThat("yup",  12.5, velocity = 3)

if __name__== "__main__":
    Pi = PiMikro()
    polPosCmos(Pi)
