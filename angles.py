import numpy as np
import sys
import argparse

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

parser.add_argument('--target','-t',type = float, help = 'target angle of the crystal in degrees', required = True)
parser.add_argument('--initial','-i',type = float, help = 'initial position angle of the crystal in degrees', required = True)

args = parser.parse_args()

target_angle = args.target
initial_angle = args.initial

delta_angle = target_angle - initial_angle

if delta_angle>0:
    step = round(np.abs(delta_angle)*10000/0.404,0)
    dir = 'Negative'
else:
    step = round(np.abs(delta_angle)*10000/0.436,0)
    dir = 'Positive'
print()
print()
print(f'    initial angle = {initial_angle} deg, target angle = {target_angle} degrees')
print(f'    Number of steps = {dir} {step}')
print()
print()
