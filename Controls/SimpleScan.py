import numpy as np
from matplotlib import pyplot as plt
import argparse

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-d', type=float,  help='Spot diameter [mm]', required=False, default=1)
parser.add_argument('-x0', type=float,  help='Spot x axis origin', required=False, default=0)
parser.add_argument('-y0', type=float,  help='Spot y axis origin', required=False, default=0)
parser.add_argument('-v', type=float,  help='Axes velocity', required=False, default=1)
args = parser.parse_args()

def ObserverOrigin():
    newX0 = args.x0 + PIx0
    newY0 = args.y0 + PIy0
    return np.array([newX0, newY0])

def NewCoord(x, y):
    O = ObserverOrigin()
    return np.array([x+O[0], y+O[1]])

def leftLine(file, point):
    file.write('2 MOV 1 %f\n'%(point[1]))
    file.write('2 WAC ONT? 1 = 1' + "\n")
    file.write('1 MOV 1 %f\n'%(point[0]))
    file.write('1 WAC ONT? 1 = 1' + "\n")
    return

def rightLine(file, point):
    file.write('2 MOV 1 %f\n'%(point[1]))
    file.write('2 WAC ONT? 1 = 1' + "\n")
    file.write('1 MOV 1 %f\n'%(point[0]))
    file.write('1 WAC ONT? 1 = 1' + "\n")
    return

def CreateSnake():
    fileName = "..\PI_MACROS\SnakeScan.txt"
    Inside = True
    plotX = np.array([])
    plotY = np.array([])
    xC = GPDl/2
    yC = GPDl/2
    with open(fileName, "w") as macro:
        macro.write('//## 1 - C-663 on USB DaisyChain: PI C-863 Mercury  SN 0022550094, device 12 (axis 1)\n')
        macro.write('//## 2 - C-663 on USB DaisyChain: PI C-863 Mercury  SN 0022550094, device 13 (axis 1)\n')
        macro.write('1 VEL 1 %f\n'%(args.v)) # setting X velocity
        macro.write('2 VEL 1 %f\n'%(args.v)) # setting X velocity
        safePos = NewCoord(xC, PIy0)
        plotX = np.append(plotX, xC)
        plotY = np.append(plotY, PIy0)
        macro.write('1 MOV 1 %f\n'%(safePos[0]))
        macro.write('1 WAC ONT? 1 = 1 ' + "\n")
        macro.write('2 MOV 1 %f\n'%(safePos[1]))
        macro.write('2 WAC ONT? 1 = 1 ' + "\n")
        while Inside:
            leftLine(macro, NewCoord(xC, yC))
            plotX = np.append(plotX, xC)
            plotY = np.append(plotY, yC)
            xC = -GPDl/2
            plotX = np.append(plotX, xC)
            plotY = np.append(plotY, yC)
            yC += -(args.d)/2
            plotX = np.append(plotX, xC)
            plotY = np.append(plotY, yC)
            rightLine(macro, NewCoord(xC, yC))
            plotX = np.append(plotX, xC)
            plotY = np.append(plotY, yC)
            xC = GPDl/2
            plotX = np.append(plotX, xC)
            plotY = np.append(plotY, yC)
            yC += -(args.d)/2
            plotX = np.append(plotX, xC)
            plotY = np.append(plotY, yC)
            if yC<-GPDl/2: Inside=False
        Guard = NewCoord(xC+2*args.d, PIy0)
        plotX = np.append(plotX, xC+2*args.d)
        plotY = np.append(plotY, yC)
        plotX = np.append(plotX, xC+2*args.d)
        plotY = np.append(plotY, PIy0)
        macro.write('1 VEL 1 3\n')
        macro.write('2 VEL 1 3\n')
        macro.write('1 MOV 1 %f\n'%(Guard[0]))
        macro.write('1 WAC ONT? 1 = 1' + "\n")
        macro.write('2 MOV 1 %f\n'%(Guard[1]))
        macro.write('2 WAC ONT? 1 = 1' + "\n")
        plt.figure()
        plt.plot(plotX, plotY, color="red")
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        plt.show()
    return

if __name__=="__main__":
    PIx0 = 12.5
    PIy0 = 12.5
    GPDl = 15.0
    CreateSnake()
