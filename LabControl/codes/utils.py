import math
import numpy as np
from astropy import units as u

def fmt_err(value, sigma, unit=None, sig=2):
    """Formatta (value +- sigma) unit con `sig` cifre significative sull'errore."""
    # Unità: quella esplicita, altrimenti quella della prima Quantity trovata
    for q in (value, sigma):
        if unit is None and isinstance(q, u.Quantity):
            unit = q.unit
    if isinstance(value, u.Quantity):
        value = value.to_value(unit)
    if isinstance(sigma, u.Quantity):
        sigma = sigma.to_value(unit)
    unit_str = f" {unit}" if unit is not None else ""

    if not np.isfinite(sigma) or sigma <= 0:
        return f"({value} +- {sigma}){unit_str}"

    nd = sig - 1 - math.floor(math.log10(abs(sigma)))
    # ricalcolo: se l'arrotondamento fa salire di ordine (0.0996 -> 0.10)
    nd = sig - 1 - math.floor(math.log10(abs(round(sigma, nd))))
    dec = max(nd, 0)
    return f"({round(value, nd):.{dec}f} +- {round(sigma, nd):.{dec}f}){unit_str}"


def LogTheta(that):
    with open("C:/Users/XCF/Desktop/PiMoveLogs/logTheta.txt", "a") as f:
        f.write(f"{that} \n")
