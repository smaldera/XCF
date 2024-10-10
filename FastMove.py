import argparse
from PiMove   import PiMikro

Pi = PiMikro()
while True:
    print('Do you need PyMicro?')
    if input()=='no':
        break
    print('Which axe should I move?')
    axe = input()
    print('Position?')
    pos = input()
    print(f'Moving {axe} in {pos}')
    Pi.MoveThat(str(axe), float(pos))
