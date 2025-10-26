import numpy as np
import matplotlib.pyplot as plt
from DataProcessing.SimData import filt_data as fd
from DataProcessing.SimData import unit_convs as uc
from DataProcessing.SimData import switching_handler as sh

dct = fd.load_filtered_sim_data_set()
fig_dpi = 300
key = 'T'

lines = (sh.switch_handle(sh.T_hl)).T_parts
# Plotting all the Data sets
plt.figure()
for i in range(3):
    plt.plot(dct[i].ssd['t'], dct[i].ssd[key], label= dct[i].name, linewidth=1)
for line in lines:
    plt.plot(dct[i].ssd['t'], line * np.ones(np.shape(dct[i].ssd['t'])), 'k--', linewidth=1)
    plt.text(1300, line+0.2, str((line*10)+200) + r'$\, ^0 C$')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel(key + uc.units[key])
plt.title("Temperature plots of Test Cell Data")
plt.savefig("./DataProcessing/SimData/figs/" + "hybrid_ssd_hl_" + key + ".png", dpi=fig_dpi)
plt.show()
