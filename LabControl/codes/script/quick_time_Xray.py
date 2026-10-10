import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================================
# CONFIGURAZIONE
# ==========================================================

DATA_DIR = "C:/Users/XCF/Documents/testMcPhersonJuly2026/23Jul"          # Cartella contenente i file .mca
EXTENSION = "*.mca"

SEARCH_MIN = 1500       # Canale minimo dove cercare il picco Ti
SEARCH_MAX = 4500       # Canale massimo

FIT_HALF_WIDTH = 80     # Canali a sinistra/destra del massimo

# ==========================================================
# MODELLO
# ==========================================================

def gauss_lin(x, A, mu, sigma, m, q):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + q


# ==========================================================
# LETTURA FILE MCA
# ==========================================================

def read_mca(filename):

    counts = []
    reading = False

    with open(filename, "r") as f:
        for line in f:

            line = line.strip()

            if line == "<<DATA>>":
                reading = True
                continue

            if line == "<<END>>":
                break

            if reading:
                try:
                    counts.append(float(line))
                except:
                    pass

    return np.array(counts)


# ==========================================================
# FIT
# ==========================================================

def fit_peak(counts):

    peak = np.argmax(counts[SEARCH_MIN:SEARCH_MAX]) + SEARCH_MIN

    xmin = max(0, peak - FIT_HALF_WIDTH)
    xmax = min(len(counts), peak + FIT_HALF_WIDTH)

    x = np.arange(xmin, xmax)
    y = counts[xmin:xmax]

    A0 = y.max() - np.median(y)
    mu0 = peak
    sigma0 = 15
    m0 = 0
    q0 = np.median(y)

    p0 = [A0, mu0, sigma0, m0, q0]

    bounds = (
        [0, peak-20, 1, -np.inf, 0],
        [np.inf, peak+20, 80, np.inf, np.inf]
    )

    popt, pcov = curve_fit(
        gauss_lin,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=100000
    )

    A, mu, sigma, m, q = popt

    area = A * sigma * np.sqrt(2*np.pi)

    return popt, area, x, y


# ==========================================================
# MAIN
# ==========================================================

files = sorted(glob.glob(os.path.join(DATA_DIR, EXTENSION)))

if len(files) == 0:
    raise RuntimeError("Nessun file .mca trovato.")

heights = []
areas = []
centers = []
sigmas = []

plt.figure(figsize=(9,6))

for i, file in enumerate(files):

    counts = read_mca(file)

    popt, area, x, y = fit_peak(counts)

    A, mu, sigma, m, q = popt

    heights.append(A)
    areas.append(area)
    centers.append(mu)
    sigmas.append(sigma)

    xx = np.linspace(x[0], x[-1], 500)

    plt.plot(
        x,
        y,
        ".",
        ms=3,
        alpha=0.35
    )

    plt.plot(
        xx,
        gauss_lin(xx, *popt),
        lw=2,
        label=f"Run {i+1}"
    )

plt.xlabel("Channel")
plt.ylabel("Counts")
plt.title("Fit Ti Kα")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()

plt.figure(figsize=(8,5))
plt.errorbar(np.arange(1,len(files)+1)*350, heights, yerr=np.sqrt(heights), lw=2)
plt.xlabel("Aq Time [s]")
plt.ylabel("Gaussian Height")
plt.grid(True)

plt.figure(figsize=(8,5))
plt.plot(np.arange(1,len(files)+1), areas, "o-", lw=2)
plt.xlabel("Run")
plt.ylabel("Gaussian Area")
plt.grid(True)

plt.figure(figsize=(8,5))
plt.plot(np.arange(1,len(files)+1), centers, "o-", lw=2)
plt.xlabel("Run")
plt.ylabel("Peak Channel")
plt.grid(True)

plt.figure(figsize=(8,5))
plt.plot(np.arange(1,len(files)+1), sigmas, "o-", lw=2)
plt.xlabel("Run")
plt.ylabel("Sigma (channels)")
plt.grid(True)

plt.show()
