import numpy as np
from astropy import constants as const
from astropy import units as u

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
        self.hc = (const.h * const.c).to(u.keV * u.femtometer)
        self.PrintInfo()

    def BraggEnergy(self, theta_deg, n=1):
        """Bragg energy at given angle (default n=1)"""
        if not isinstance(theta_deg, u.Quantity):
            theta_deg = theta_deg * u.deg
        theta = np.radians(theta_deg.to(u.deg).value) * u.rad
        lambda_bragg = self.p_fm * np.sin(theta)
        return n * self.hc / lambda_bragg

    def BraggTheta(self, Energy):
        """Angle needed for target energy (inverse of BraggEnergy)"""
        if not isinstance(Energy, u.Quantity):
            Energy = Energy * u.keV
        sinTheta = self.hc / (self.p_fm * Energy)
        return np.arcsin(sinTheta).to(u.deg)

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

    def sigmaBraggTheta(self, Energy, sigmaEnergy):
        if not isinstance(Energy, u.Quantity):
            Energy = Energy * u.keV
        if not isinstance(sigmaEnergy, u.Quantity):
            sigmaEnergy = sigmaEnergy * u.keV
        var = 1/(((self.p_fm*Energy/self.hc))**2 - 1)*(sigmaEnergy/Energy)**2
        sigma_rad = np.sqrt(var).to_value(u.dimensionless_unscaled) * u.rad
        return sigma_rad.to(u.deg)
