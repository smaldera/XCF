import numpy as np
from scipy.optimize import curve_fit

class GF1():
    # The function must be a string like: "[0] + [1]*x"
    def __init__(self, expression):
        self.function = expression
        self.nPar = 0
        self.dict = {
            'exp':  np.exp,
            'log':  np.log,
            'sin':  np.sin,
            'cos':  np.cos,
            'tan':  np.tan,
            'pi':   np.pi,
            'sqrt': np.sqrt,
        }

    def __call__(self, x, *parameters):
        modify = self.function
        self.dict['x'] = x
        for i, value in enumerate(parameters):
            modify = modify.replace(f'[{i}]', str(value))
            self.nPar = i
        return self.Eval(modify)
        
    def Eval(self, function):
        try:
            return eval(function, {"__builtins__": None}, self.dict)
        except Exception as e:
            raise ValueError(f"Evaluation error: {e}")

    def GetNumberOfParameters(self):
        return self.nPar

        
class GFit():
    # Delta sets the minimum relative difference between Chi2 obtained with iterating fits
    # MaxCycle sets the maximum number of iterationsepsilon
    def __init__(self, expression, delta = 0.001, MaxCycle = 5):
        print(f'New Function added: {expression}')
        self.GF = GF1(expression)
        self.delta = delta
        self.nMax = MaxCycle
        self.X2 = None

    def __call__(self, x, y, errY=0., params=None, bounds=None):
        print('-'*100 + f'\nRepeated Fit enabled, maximum number of cycles set to {self.nMax}')
        epsilon = 0.
        n = 0
        if bounds is None:
            bounds = self.InfBounds()
        while True:
            if params is None:
                params, covariance = curve_fit(self.GF, x, y, sigma=errY, bounds=bounds)
            else:
                params, covariance = curve_fit(self.GF, x, y, sigma=errY, p0=params, bounds=bounds)
            print(f'Cycle: {n}, Parameters: {params}')
            Chi2 = self.ChiSquare(x, y, params)
            print(f'Chi2: {Chi2} with {len(x)-len(params)} dof.\n Reduced Chi2: {Chi2/(len(x)-len(params))}')
            if (np.abs(epsilon-Chi2)  < self.delta) | (n > self.nMax):
                print('-'*100)
                return params, covariance
            epsilon = Chi2
            n += 1

    def ChiSquare(self, x, O_i, params):
        E_i = np.array(self.GF(x, *params))
        Chi_i = ( (O_i - E_i)**2 )/E_i
        self.X2 = Chi_i.sum()
        return Chi_i.sum()

    def GetChi2(self):
        return self.X2

    def GetFunction(self):
        return self.GF

    def InfBounds(self):
        n_params = self.GF.GetNumberOfParameters()
        bounds = ([-np.inf] * n_params, [np.inf] * n_params)
        return bounds
