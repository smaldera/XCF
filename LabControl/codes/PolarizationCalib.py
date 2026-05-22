"""
This script is made by XCF team to correctly position the crystal inside the chamber.
It cycles trough a routine of energy measurements using the SDD and repositioning the crystal.
At the end it will save everything in one file.
"""

import argparse as ar
import numpy as np
from AmptekControl import SDD
from astropy import constants as const
from astropy import units as u
from PiMove import PiMikro

formatter = ar.ArgumentDefaultsHelpFormatter
parser = ar.ArgumentParser(formatter_class=formatter)
parser.add_argument('-target', type=float, help='target angle', required=True)
parser.add_argument('-path', type=str, help='npz file path', default='./', required=False)
parser.add_argument('-suf', type=str, help='saved file name suffix', default='', required=False)
parser.add_argument('-crEl', type=str, help='crystal element: InSb111, Ge111, Si111, Si220, Si400, Ge422', default='InSb111', required=False)
parser.add_argument('-time', type=float, help='livetime', default=180., required=False)
parser.add_argument('-sigma', type=float, help='Energy difference from target [keV]', default=0.002, required=False)
args = parser.parse_args()

hc = (const.h * const.c).to(u.keV * u.femtometer)
def has_unit(variable):
    return hasattr(variable, 'unit')

class Crystal(object):
    def __init__(self, element):
        self.crystal_vocab = {
            'InSb111': {'phase': 7.481e5, 'energy': 2.293, 'theta':46.28, 'anode':'Mo'},
            'Ge111':   {'phase': 6.532e5, 'energy': 2.697, 'theta':44.87, 'anode':'Rh'},
            'Si111':   {'phase': 6.271e5, 'energy': 2.839, 'theta':44.12, 'anode':'Pd'},
            'Si220':   {'phase': 3.840e5, 'energy': 4.511, 'theta':45.71, 'anode':'Ti'}, 
            'Si400':   {'phase': 2.714e5, 'energy': 6.400, 'theta':45.60, 'anode':'Fe'},
            'Si400':   {'phase': 2.310e5, 'energy': 7.478, 'theta':45.86, 'anode':'Ni'},
        }
        if element not in self.crystal_vocab:
            raise ValueError(f"Element '{element}' not available.")
        self.element = element
        self.p_fm = self.crystal_vocab[element]['phase'] * u.femtometer
        self.energy = self.crystal_vocab[element]['energy'] * u.keV
        self.theta = self.crystal_vocab[element]['theta'] * u.deg
        self.anode = self.crystal_vocab[element]['anode']
        self.PrintInfo()
        
    def BraggEnergy(self, theta_deg, n=1):
        if has_unit(theta_deg)==False:
            theta_deg = theta_deg * u.deg
        theta = np.radians(theta_deg.to(u.deg).value) * u.rad
        lambda_bragg = (self.p_fm * np.sin(theta))
        return n*hc / lambda_bragg
    
    def BraggTheta(self, Energy):
        if has_unit(Energy)==False:
            Energy = Energy * u.keV
        sinTheta = hc/(self.p_fm*Energy)
        return np.degrees(np.arcsin(sinTheta))
    
    def Get2P(self):
        return self.p_fm
    
    def GetEnergy(self):
        return self.energy
    
    def GetTheta(self):
        return self.theta
    
    def PrintInfo(self):
        print(f'Crystal selected succesfully: {self.element}')
        print(f' - 2 * crystal phase: {self.p_fm}')
        print(f' - Energy selected at 45°: {np.round(self.BraggEnergy(45.), 2)}')
        print(f' - Angle used for the coupled anode ({self.anode}): {np.round(self.BraggTheta(self.energy), 2)}\n')

if __name__== "__main__":
    cr = Crystal(args.crEl)
    if args.path == './':
        path = '/mnt/c/Users/XCF/'
    else:
        path = '/mnt/c/Users/XCF/' + args.path
    print(f'Starting Calibration routine. Each calibration cycle will take {args.time} seconds.')
    sigma = (args.sigma * u.keV).to(u.eV)

    Amptek = SDD()
    Pi = PiMikro()
    while True:
        name = args.name + str(i)
        # in data are stored the bins heigth of the mca spectrum
        data, utilData = Amptek.SaveAndAcquire(livetime=args.time, path=path, name=args.name)
        # searches for the mode value
        chn = np.linspace(0, len(data))
        mode_energy = SDD.Chn2keV(chn[np.where(data == data.max())]) * u.keV
        print(f'Measured energy: {mode_energy}.')
        print(f'Current crystal angle: {cr.BraggTheta(mode_energy)}')
        delta = (np.abs(mode_energy-cr.BraggEnergy(args.target))).to(u.eV)
        print(f'Energy delta {delta}.')
        # checks that the mode value is the same as we expected
        if delta <= sigma:
            print(f'The difference in angle is acceptable.\n')
            print(f'FINAL ANGLE: {cr.BraggTheta(mode_energy)}')
            break
        # CODE 2 GET THAT THING CHANGE THETA
        i += 1