from pipython import GCSDevice, pitools
import numpy as np
from tqdm import tqdm
from Point import Point2D
import time

"""Note that all tracks generators have games names"""

class PiMikro:
    def __init__(self, comport=4, baudrate=115200, seed=None):
        self.comport = comport
        self.baudrate = baudrate
        self.GPDl = 15 #GPD frame length [mm]
        self.upRange = 25 #[mm]
        self.xOrigin = 0 #[mm]
        self.yOrigin = 0 #[mm]
        self.hCrystal = GCSDevice()
        self.hCrystal.OpenRS232DaisyChain(comport=self.comport, baudrate=self.baudrate)
        daisychainid = self.hCrystal.dcid
        self.hCrystal.ConnectDaisyChainDevice(1, daisychainid)
        self.xLow = GCSDevice()
        self.xLow.ConnectDaisyChainDevice(8, daisychainid)
        self.yUp = GCSDevice()
        self.yUp.ConnectDaisyChainDevice(12, daisychainid)
        self.xUp = GCSDevice()
        self.xUp.ConnectDaisyChainDevice(13, daisychainid)
        self.yLow = GCSDevice()
        self.yLow.ConnectDaisyChainDevice(14, daisychainid)
        self.piAngle = GCSDevice()
        self.piAngle.ConnectDaisyChainDevice(15, daisychainid)
        self.seed = 0
        if seed is None:
            self.seed = int(time.time())
            np.random.seed(int(time.time()))
        else:
            self.seed = seed
            np.random.seed(seed)
        print('Connection status: SUCCESS')
        print(f'Random seed set to: {self.seed}')

    def Cobra(self, velocity=1, step = 0.5, amplitude=7, Pa=900, Px=107, Py=127, Nstep=10000):
        """This fuction emulates dithering movements"""
        k = 0
        self.xUp.VEL(1, velocity)
        self.yUp.VEL(1, velocity)
        Origin = Point2D()
        Origin = self.ObserverOrigin()
        while True:
            goTo = self.ObserverOrigin()
            print('\n', k)
            print("(", goTo.GetX(), ", ", goTo.GetY(), ")")
            goTo.Shift(amplitude*self.xDithering(Pa, Px, k), amplitude*self.yDithering(Pa, Py, k))
            self.fastReach(goTo)
            if k >= Nstep:
                break
            k+=step
        return 1

    def fastReach(self, Target, verbose=False):
        """Given a point it moves to it using Up axe"""
        if verbose:
            print(f'fastReach Target: {Target.GetX()} , {Target.GetY()}')
        self.xUp.MOV(1, Target.GetX())
        self.yUp.MOV(1, Target.GetY())
        pitools.waitontarget(self.xUp, axes=1)
        pitools.waitontarget(self.yUp, axes=1)
        return 1

    def GaussQuake(self, width=0.5):
        # [width] = mm !!
        # [stop] = s !!
        P0 = self.ObserverOrigin()
        P0.Shift(np.random.normal(scale=width, loc=0.), np.random.normal(scale=width, loc=0.))
        self.MoveOn(P0)
        return 1

    def GetSeed(self):
        return self.seed

    def HotSpot(self, x0, y0):
        """Sets the coordinate origin on the hotspot (GPD axes)"""
        self.xOrigin = x0 #[mm]
        self.yOrigin = y0 #[mm]
        return 1

    def Mamba(self, side, d): #It doesn't work like that: you need a np.array() of different Points
        """Given the side's lenght of the square and the 2*spacing
           between the lines, it returns the track"""
        target = np.empty((1,2))
        pos = Point2D()
        pos = self.NewCoord(side, side)
        y = pos.GetY()
        while True:
            target = np.append(target, pos.GetCoord(), axis=0)
            if y<self.NewCoord(0,-side).GetY():
                break
            y += -d/2
            pos.Shift(-2*side, -d/2)
            target = np.append(target, pos.GetCoord(), axis=0)
            y += -d/2
            pos.Shift(2*side, -d/2)
        target = target[1:]
        snake = np.concatenate((target, target[::-1]))
        print('\nMamba done\n')
        print(snake)
        return snake

    def MoveOn(self, target):
        """Given a point it moves to it using Up axe WITHOUT waiting"""
        self.xUp.MOV(1, target.GetX())
        self.yUp.MOV(1, target.GetY())
        return 1

    def MoveThat(self, axe, pos, velocity = 1):
        if axe == 'xup':
            self.xUp.VEL(1, velocity)
            self.xUp.MOV(1, pos)
            pitools.waitontarget(self.xUp, axes=1)
        elif axe == 'yup':
            self.yUp.VEL(1, velocity)
            self.yUp.MOV(1, pos)
            pitools.waitontarget(self.yUp, axes=1)
        elif axe == 'xlow':
            self.xLow.VEL(1, velocity)
            self.xLow.MOV(1, pos)
            pitools.waitontarget(self.xLow, axes=1)
        elif axe == 'ylow':
            self.yLow.VEL(1, velocity)
            self.yLow.MOV(1, pos)
            pitools.waitontarget(self.yLow, axes=1)
        elif axe == 'h':
            self.hCrystal.VEL(1, velocity)
            self.hCrystal.MOV(1, pos)
            pitools.waitontarget(self.hCrystal, axes=1)
        elif axe == 'rot':
            self.piAngle.VEL(1, velocity)
            self.piAngle.MOV(1,pos)
            pitools.waitontarget(self.piAngle, axes=1)
        return 1

    def MoveTheSnake(self, nLoops, snake, velocity): #It doesn't work like that: you need to read an array of points
        """Makes the snake move in position with a given velocity for nLoops times"""
        self.xUp.VEL(1, velocity)
        self.yUp.VEL(1, velocity)
        for i in range(nLoops):
            for corner in tqdm(snake, desc='Snake %i'%(i)):
                self.xUp.MOV(1, corner[0])
                pitools.waitontarget(self.xUp, axes=1)
                self.yUp.MOV(1, corner[1])
                pitools.waitontarget(self.yUp, axes=1)
        return 1

    def NewCoord(self, x, y):
        """Given a point returns its new coordinate"""
        O = self.ObserverOrigin()
        return Point2D(x+O.GetX(), y+O.GetY())

    def ObserverOrigin(self):
        """Shifts the origins in (x0,y0)"""
        newX0 = self.xOrigin + self.upRange/2
        newY0 = self.yOrigin + self.upRange/2
        return Point2D(newX0, newY0)

    def Rattling(self, nLoops, snake, velocity, direction='y', maxShift=0.5):
        """Makes the snake move in position with a given velocity for nLoops times.
           This makes the snake RATTLE which means shaking or shifting its positions"""
        self.xUp.VEL(1, velocity)
        self.yUp.VEL(1, velocity)
        for i in range(nLoops):
            rattle = np.random.rand()*maxShift
            #print('Rattle: ', rattle)
            for corner in tqdm(snake, desc='Snake %i'%(i)):
                if direction=='y':
                    self.yUp.MOV(1, corner[1]+rattle)
                    pitools.waitontarget(self.yUp, axes=1)
                    self.xUp.MOV(1, corner[0])
                    pitools.waitontarget(self.xUp, axes=1)
                elif direction=='xy':
                    self.yUp.MOV(1, corner[1]+rattle)
                    pitools.waitontarget(self.yUp, axes=1)
                    self.xUp.MOV(1, corner[0]+rattle)
                    pitools.waitontarget(self.xUp, axes=1)
                elif direction=='x':
                    self.yUp.MOV(1, corner[1])
                    pitools.waitontarget(self.yUp, axes=1)
                    self.xUp.MOV(1, corner[0]+rattle)
                    pitools.waitontarget(self.xUp, axes=1)
                else:
                    print('Invalid direction: use x, y or xy')
                    return 1
        return 1

    def ReactionTime(self, axe, pos, velocity):
        start = time.time()
        self.MoveThat(axe, pos, velocity=velocity)
        return time.time()-start

    def SpotPosition(self):
        self.xLow.MOV(1, 28.15)
        self.yLow.MOV(1, 11.25)
        self.xUp.MOV(1, 12.5)
        self.yUp.MOV(1, 12.5)
        pitools.waitontarget(self.xLow, axes=1)
        pitools.waitontarget(self.yLow, axes=1)
        pitools.waitontarget(self.xUp, axes=1)
        pitools.waitontarget(self.yUp, axes=1)
        return 1

    def Sleeper(self, sleeptime):
        time.sleep(sleeptime)
        return 'Go'

    def UniformQuake(self, width=0.5, P0 = Point2D(), verbose=False):
        # [width] = mm !!
        # [stop] = s !!
        if verbose:
            print(f'EpiCenter : {P0.GetX()} , {P0.GetY()}')
        P0.Shift(np.random.uniform(low=-width, high=width), np.random.uniform(low=-width, high=width))
        if verbose:
            print(f'Shifted in: {P0.GetX()} , {P0.GetY()}')
        self.fastReach(P0)
        return 1

    def UnPolPosition(self, deg=None):
        Plow = Point2D()
        Pup = Point2D()
        if deg == -90:
            print(f'Degree: {deg}')
            Plow.Shift(28.5, 11.)
            Pup.Shift(12.5, 12.5)
        if deg == 0:
            print(f'Degree: {deg}')
            Plow.Shift(40.0, 0.)
            Pup.Shift(12.5, 20.8)
        else:
            print('! Degree not specified !')
            return 1, 1
        self.xLow.MOV(1, Plow.GetX())
        self.yLow.MOV(1, Plow.GetY())
        self.xUp.MOV(1, Pup.GetX())
        self.yUp.MOV(1, Pup.GetY())
        pitools.waitontarget(self.xLow, axes=1)
        pitools.waitontarget(self.yLow, axes=1)
        pitools.waitontarget(self.xUp, axes=1)
        pitools.waitontarget(self.yUp, axes=1)
        return Plow, Pup

    def upLine(self, A, B):
        """Given 2 points, moves to B by A using Up axes.
           It does it by moving in one direction at a time"""
        if self.xUp.qPOS != A.GetX():
            self.xUp.MOV(1, A.GetX())
            pitools.waitontarget(self.xUp, axes=1)
        if self.yUp.qPOS != A.GetY():
            self.yUp.MOV(1, A.GetY())
            pitools.waitontarget(self.yUp, axes=1)
        self.xUp.MOV(1, B.GetX())
        pitools.waitontarget(self.xUp, axes=1)
        self.yUp.MOV(1, B.GetY())
        pitools.waitontarget(self.yUp, axes=1)
        return 1

    def upReach(self, Target):
        """Given 1 points, moves to it using Up axes.
           It moves in one direction at a time"""
        self.xUp.MOV(1, Target.GetX())
        pitools.waitontarget(self.xUp, axes=1)
        self.yUp.MOV(1, Target.GetY())
        pitools.waitontarget(self.yUp, axes=1)
        return 1

    def xDithering(self, a, x, k):
        """Returns x as in Dithering function evolving with k"""
        return np.cos(2*np.pi*k/a)*np.cos(2*np.pi*k/x + np.pi/2)

    def yDithering(self, a, y, k):
        """Returns y as in Dithering function evolving with k"""
        return np.sin(2*np.pi*k/a)*np.sin(2*np.pi*k/y)
