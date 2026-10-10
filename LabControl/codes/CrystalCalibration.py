import argparse as ar
from astropy import units as u
from Crystal import Crystal
import numpy as np
from PiMove   import PiMikro
from SDDscan import StandardScan, MeasureEnergy, ScanLineSearcher
from utils import fmt_err, LogTheta
import time

formatter = ar.ArgumentDefaultsHelpFormatter
parser = ar.ArgumentParser(formatter_class=formatter)
parser.add_argument('-target', type=float, help='target angle', required=True)
parser.add_argument('-crEl', type=str, help='crystal element: InSb111, Ge111, Si111, Si220, Si400, Ge422', default='InSb111', required=False)
args = parser.parse_args()
args.target = args.target * u.deg

#Istantiate crystal
Cr = Crystal(args.crEl)

#Connect to SDD
Pi = PiMikro()
t0 = time.time()

rate_array = []

delta_angle = 0.2* u.deg

#step0 = Pi.Angle2Step(0, 0-3*delta_angle)
step0 = Pi.Angle2Step(0, 0-5*delta_angle)
Pi.MoveTheta(step0)
p0, r0 = ScanLineSearcher(Pi)
rate_array.append(r0)
i=0

#for i in range(0,6):
for i in range(0,10):
    Pi.MoveTheta(Pi.Angle2Step(0, delta_angle))
    p, r = ScanLineSearcher(Pi)
    rate_array.append(r)
    print(f"fast scan... i ={i} Max rate = {r} Hz, Position = {p} mm")


#rate_array=[1.2, 1.0, 1.2666666666666666, 1.0666666666666667, 1.3333333333333333, 1.4, 3.066666666666667, 58.06666666666667, 169.93333333333334, 247.46666666666667, 235.66666666666666]
print(rate_array)
max_angle_rate = np.max(rate_array)
index = np.where(rate_array == max_angle_rate)[0][0]
print(index)
Pi.MoveTheta(Pi.Angle2Step(0, -(len(rate_array)-index -1) * delta_angle))

p, r = ScanLineSearcher(Pi)
print(f"Max rate = {r} Hz, Position = {p} mm")
print("------------------------------------------------------------------------------")

delta_angle_2 = 0.05 *u.deg

step0_2 = Pi.Angle2Step(0, 0-5*delta_angle_2)
Pi.MoveTheta(step0_2)
i=0

rate_x = []
rate_y = []
pos_x = []
pos_y = []
posStep = 0
negStep = 0
for i in range(0,9):
    step = Pi.Angle2Step(0, delta_angle_2)
    Pi.MoveTheta(step)
    if step > 0:
        posStep = posStep + step
    else:
        negStep = negStep + step
    LogTheta(step)
    scan, xLow_max_rate, max_rate_x, yLow_max_rate, max_rate_y = StandardScan(Pi, xLow_max = 93, xLow_min = 79, yLow_min = 5, yLow_max = 19)
    #if scan == False:
    #    print("Max rate below threshold, the line is lost")
    #    print(f"posStep = {posStep}, negStep = {negStep}")
    #    delta_angle_while = Pi.Step2Angle(posStep) + Pi.Step2Angle(negStep)
    #    step_back = -Pi.Angle2Step(0, delta_angle_while)
    #    Pi.MoveTheta(step_back)
    #    print(f"Angle moved by {step_back} step")
    #    break
    if scan==True:
        rate_x.append(max_rate_x)
        rate_y.append(max_rate_y)
        pos_x.append(xLow_max_rate)
        pos_y.append(yLow_max_rate)
    print(f"standatrd scan... i= {i} Max rate x = {max_rate_x} Hz, Position x = {xLow_max_rate} mm, Max rate y = {max_rate_y} Hz, Position y = {yLow_max_rate} mm")
if len(rate_y) == 0:
    print("Rate to low on every position/angle over the scan range.")
    exit()
max_angle_rate_y = np.max(rate_y)


#qua controlla se ha perso la linea...  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if max_angle_rate_y < 5:
    print("Max rate below threshold, the line is lost")
    print(f"posStep = {posStep}, negStep = {negStep}")
    delta_angle_while =(Pi.Step2Angle(posStep) + Pi.Step2Angle(negStep) )*u.deg
    step_back = -Pi.Angle2Step(0, delta_angle_while)

    Pi.MoveTheta(step_back)
    print(f"Angle moved by {step_back} step")
    print("aborting... ")
    exit()



index_2 = np.where(rate_y == max_angle_rate_y)[0][0]
print("indice max:", index_2)
Pi.MoveTheta(Pi.Angle2Step(0, -(len(rate_y)-index_2 -1) * delta_angle_2))
Pi.MoveThat(axe='xlow', pos=pos_x[index_2]) ############################################ !!!!
print(f"Moved xlow to {pos_x[index_2]} mm")
Pi.MoveThat(axe='ylow', pos=pos_y[index_2]) ############################################ !!!!
print(f"Moved ylow to {pos_y[index_2]} mm")

# Max rate x = 250.8 Hz, Position x = 86,4288868013164 mm, Max rate y = 264.53333333333336 Hz, Position y = 10,5 mm





"""
#exit()
posStep = 0
negStep = 0
#Main Loop
while True:
    #Scan for the max rate (line center)
        = StandardScan(Pi)
    if scan == False:
        print("Max rate below threshold, the line is lost")
        print(f"posStep = {posStep}, negStep = {negStep}")
        delta_angle_while = Pi.Step2Angle(posStep) + Pi.Step2Angle(negStep)
        step_back = -Pi.Angle2Step(0, delta_angle_while)
        Pi.MoveTheta(step_back)
        print(f"Angle moved by {step_back} step")
        break
    t1 = time.time()
    print(f"Scan ended in {np.round(t1 - t0, 1)} s")
    #Energy measurment
    energy, sigmaEnergy = MeasureEnergy(Pi)
    #Energy conversion to angle (degree)
    angle = Cr.BraggTheta(energy)
    sigmaAngle = Cr.sigmaBraggTheta(energy, sigmaEnergy)
    print(f"Detected energy: {fmt_err(energy, sigmaEnergy, 'keV')}")
    print(f"Conversion to angle: {fmt_err(angle, sigmaAngle, 'deg')}\n")
    #Move to target Angle
    if (np.abs(angle - args.target) < sigmaAngle ):
        print("Target angle reached.")
        LogTheta("----------------")
        break
    step = Pi.Angle2Step(angle, args.target)
    #breakpoint()
    Pi.MoveTheta(step)
    if step > 0:
        posStep = posStep + step
    else:
        negStep = negStep + step
    LogTheta(step)
    print(f"Angle moved in {np.round(time.time() - t1, 1)} s")
    t0 = time.time()


"""
