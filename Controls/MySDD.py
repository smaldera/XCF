import numpy as np
import time
import sys
import amptek_hardware_interface as Amp
from matplotlib import pyplot as plt

SDD = Amp.AmptekHardwareInterface()
SDD.connectUSB(-1)

if SDD.Ping():
   print('Ping succeeded')
else:
    print('Ping failed!')
    sys.exit()

config_names = ["RESC", "CLCK", "TPEA", "GAIF", "GAIN", "RESL", "TFLA", "TPFA", 
               "PURE", "RTDE", "MCAS", "MCAC", "SOFF", "AINP", "INOF", "GAIA",
               "CUSP", "PDMD", "THSL", "TLLD", "THFA", "DACO", "DACF", "DACF",
               "RTDS", "RTDT", "BLRM", "BLRD", "BLRU", "AUO1", "PRET", "PRER",
               "PREC", "PRCL", "PRCH", "HVSE", "TECS", "PAPZ", "PAPS", "SCOE",
               "SCOT", "SCOG", "MCSL", "MCSH", "MCST", "AUO2", "TPMO", "GPED",
               "GPGA", "GPMC", "MCAE", "VOLU", "CON1", "CON2"]
configs = SDD.GetTextConfiguration( config_names )

print("----CURRENT CONFIG-----")
for config in configs:
    print(config)
print("-----------------------\n\n")

SDD.ClearSpectrum()
SDD.SetPresetAccumulationTime(100)
    
SDD.Enable()
print("Acquisition started")

while True:
    time.sleep(1)    
    status = SDD.updateStatus(-1)
    print("\rAccumulation Time: {:.2f}s, Fast Counts: {:d}, Slow Counts: {:d}".format( status.AccTime(), status.FastCount(), status.SlowCount() ), end="", flush=True)

    # test if finished
    if not status.IsEnabled():
        print("")
        break 

print("Acquisition finished")

data = np.array(SDD.GetSpectrum())
fig, ax = plt.subplots()
ax.plot( data )
plt.show()
