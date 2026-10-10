import argparse
import numpy as np
import os
from PiMove   import PiMikro
from Point    import Point2D
import time
import ConnectToWSL as WSL


def GradientSDDrate(Pi: PiMikro, axe, coord_min, coord_max, step=1):
    """
    Adaptive scan: coarse grid → find gradient sign change →
    fine refinement around maximum
    """
    print(f"\n[PHASE 1] Coarse scan ({axe})")

    # Phase 1: Coarse scan with large steps (1.5 mm)
    positions_coarse = np.arange(coord_min, coord_max + 1, 1.5)
    rates_coarse = MoveAndRate(positions_coarse)

    # Find where gradient changes sign (coarse maximum)
    gradients = np.diff(rates_coarse)
    sign_changes = np.where(np.diff(np.sign(gradients)) != 0)[0]

    if len(sign_changes) == 0:
        # No local maximum found, use global max
        max_idx = np.argmax(rates_coarse)
        center_pos = positions_coarse[max_idx]
        print(f"  No gradient sign change: using global max at {center_pos} mm")
    else:
        # Found maximum region (most likely the first one)
        max_idx = sign_changes[0]
        center_pos = positions_coarse[max_idx]
        print(f"  Gradient sign change at index {max_idx}: center={center_pos} mm")

    print(f"\n[PHASE 2] Fine scan around {center_pos} mm")

    # Phase 2: Fine scan around maximum (0.5 mm steps, ±3 mm window)
    fine_margin = 3  # mm
    fine_step = 0.5  # mm (or 1 if too slow)
    positions_fine = np.arange(
        center_pos - fine_margin,
        center_pos + fine_margin + fine_step,
        fine_step
    )

    rates_fine = MoveAndRate(positions_fine)
    positions_fine = np.array(positions_fine)

    # Find exact maximum
    max_idx_fine = np.argmax(rates_fine)
    opt_pos = positions_fine[max_idx_fine]
    opt_rate = rates_fine[max_idx_fine]

    # Bonus: fit parabola to find sub-step precision
    try:
        if max_idx_fine > 0 and max_idx_fine < len(rates_fine) - 1:
            opt_rate = fit4Max(positions_fine, rates_fine)
    except:
        print(f"  Parabolic fit failed, using grid maximum")

    print(f"\nFinal optimum: {axe}={opt_pos:.2f}mm ({opt_rate:.1f} Hz)")
    Pi.MoveThat(axe, opt_pos)
    return opt_pos

def fit4Max(position, rate):
    """
    Fit parabola around peak
    """
    max_rate_id = np.where(rate==np.max(rate))[0][0]
    rate_mask = [max_rate_id-2, max_rate_id-1, max_rate_id, max_rate_id+1, max_rate_id+2]
    print(f"Parabolic range = {position[rate_mask]}")
    coeffs = np.polyfit(position[rate_mask], rate[rate_mask], 2)
    opt_pos_refined = -coeffs[1] / (2 * coeffs[0])  # Vertex formula
    print(f"  Parabolic fit: optimum = {opt_pos_refined:.2f}mm")
    return opt_pos_refined

def MeasureEnergy(Pi: PiMikro):
    energy_array = []
    WSL.callps1(path="SDDmove/", livetime='120', name=f'energy', function=f'e')
    with open("C:/Users/XCF/SDDmove/energy.txt", "r") as f:
         values = [float(row.strip()) for row in f if row.strip()]
         energy = values[1]
         sigmaEnergy = np.sqrt(values[4])
         #sigmaEnergy = values[2]
         print(f"Line energy = {energy} ± {sigmaEnergy} [keV]")
    return energy, sigmaEnergy

def MoveAndRate(position):
    rates_array = []
    for pos in positions_coarse:
        Pi.MoveThat(axe, pos)
        rate = get_rate_at_position(Pi)
        rates_array.append(rate)
        print(f"  {axe}={pos:3d}mm: {rate:7.1f} Hz")
    return np.array(rates_array)

def MoveSDDRate(Pi: PiMikro, axe, coord, coord_max, step = 1):
    rate_array = []
    coord_array = []

    while coord <= coord_max:
        Pi.MoveThat(axe,coord)
        print(f"Moved {axe} to {coord} mm")
        coord_array.append(coord)
        #print(coord_array)
        WSL.callps1(path="SDDmove/", name=f'rate', function=f'r')
        with open("C:/Users/XCF/SDDmove/rate.txt", "r") as f:
            rate = float(f.read().strip())
            print(f"Measured rate: {rate} Hz")
            rate_array.append(rate)
        f.close()
        #print(rate_array)
        Pi.Sleeper(2)
        coord = coord + step

    rate_array = np.array(rate_array)
    coord_array = np.array(coord_array)
    max_rate = np.max(rate_array)
    if max_rate > 5:
        coord_max_rate = fit4Max(coord_array, rate_array)
        Pi.MoveThat(axe,coord_max_rate)
        print(f"Final point ({axe}) set to {coord_max_rate} mm \n")
    else:
        coord_max_rate = None
    return coord_max_rate, max_rate

def MoveSDDRateLineSearcher(Pi: PiMikro, axe, coord, coord_max):
    rate_array = []
    coord_array = []

    while coord <= coord_max:
        Pi.MoveThat(axe,coord)
        print(f"Moved {axe} to {coord} mm")
        coord_array.append(coord)
        #print(coord_array)
        WSL.callps1(path="SDDmove/", name=f'rate', function=f'r')
        with open("C:/Users/XCF/SDDmove/rate.txt", "r") as f:
            rate = float(f.read().strip())
            print(f"Measured rate: {rate} Hz")
            rate_array.append(rate)
        f.close()
        #print(rate_array)
        Pi.Sleeper(2)
        coord = coord + 2.5

    rate_array = np.array(rate_array)
    coord_array = np.array(coord_array)
    max_rate = np.max(rate_array)
    coord_max_rate = coord_array[np.where(rate_array==max_rate)[0]][0]
    Pi.MoveThat(axe,coord_max_rate)
    print(f"Final point ({axe}) set to {coord_max_rate} mm \n")
    return coord_max_rate, max_rate

def ScanLineSearcher(Pi: PiMikro, xLow_max = 91, xLow_min = 81, yLow_min = 3, yLow_max = 21, yLow0 = 12, yUp = 18, xUp = 12):
    print(f"-----------------------------------------------")
    print(f"Starting SDD max rate scan.")
    PiLow0 = Point2D(xLow_min,yLow0)
    PiUp0 = Point2D(xUp,yUp)
    Pi.fastReach(PiUp0)
    Pi.LowReach(PiLow0)
    print(f"Initial position: Low ({xLow_min} , {yLow0}) ;  Up ({xUp} , {yUp})")

    xLow_max_rate, rate_max = MoveSDDRateLineSearcher(Pi, axe='xlow', coord=xLow_min, coord_max=xLow_max)
    #yLow_max_rate = MoveSDDRateLineSearcher(Pi, axe='ylow', coord=yLow_min, coord_max=yLow_max)
    print(f"-----------------------------------------------")
    return xLow_max_rate, rate_max


def StandardScan(Pi: PiMikro, xLow_max = 95, xLow_min = 77, yLow_min = 3, yLow_max = 21, yLow0 = 12, yUp = 18, xUp = 12):
    print(f"-----------------------------------------------")
    print(f"Starting SDD max rate scan.")
    PiLow0 = Point2D(xLow_min,yLow0)
    PiUp0 = Point2D(xUp,yUp)
    Pi.fastReach(PiUp0)
    Pi.LowReach(PiLow0)
    print(f"Initial position: Low ({xLow_min} , {yLow0}) ;  Up ({xUp} , {yUp})")

    xLow_max_rate, max_rate_x = MoveSDDRate(Pi, axe='xlow', coord=xLow_min, coord_max=xLow_max)
    if xLow_max_rate is None:
        return False, None, None, None, None
    else:
        yLow_max_rate, max_rate_y = MoveSDDRate(Pi, axe='ylow', coord=yLow_min, coord_max=yLow_max)
        print(f"-----------------------------------------------")
        return True, xLow_max_rate, max_rate_x, yLow_max_rate, max_rate_y

if __name__== "__main__":
    Pi = PiMikro()
    StandardScan(Pi)
