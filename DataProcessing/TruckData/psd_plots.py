import numpy as np
import rdRawDat as rd
import filt_data as fd
import psd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkAgg')

# ===============================================================================================
raw_dat = rd.load_truck_data_set()
filt_dat = fd.load_filtered_truck_data_set()


states = ['u1', 'u2', 'T', 'F', 'eta', 'y1']
dat = filt_dat
for age in range(2):
    for test in range(3):
        for state in states:
            plt.figure(state)
            time_series = dat[age][test].iod[state]
            tskips = dat[age][test].iod['t_skips']
            fs = 1 / dat[age][test].dt
            f, mag = psd.welch_TD(time_series, tskips, fs)
            plt.plot(f, mag, label=dat[age][test].name, linewidth=1)
            plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
            plt.xlim([-0.01, 0.2])
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude [normalized]')
            plt.grid(True)
            plt.legend(loc='best')

plt.show()
