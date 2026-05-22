import numpy as np

class Point2D:
    def __init__(self, X=0., Y=0.):
        self.coord = np.array([X, Y])

    def SetX(self, newX):
        self.coord[0] = newX
        return 1

    def SetY(self, newY):
        self.coord[1] = newY
        return 1

    def SetPoint(self, X, Y):
        self.SetX(X)
        self.SetY(Y)
        return 1

    def GetX(self):
        return self.coord[0]

    def GetY(self):
        return self.coord[1]

    def Shift(self, addX, addY):
        self.SetPoint(self.GetX()+addX, self.GetY()+addY)

    def GetCoord(self):
        return np.array([[self.GetX(), self.GetY()]])
