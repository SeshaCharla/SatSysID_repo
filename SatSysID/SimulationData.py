import numpy as np
from DataProcessing.SimData import filt_data
from DataProcessing.SimData import unit_convs as uc
import matplotlib.pyplot as plt
from SatSysID import SatSysID_funcs as sf
from SatSysID.SatSysID_methods import SatSys_ssd

# =====================================================

sim_data = filt_data.load_filtered_sim_data_set()
# tst_data = decimate_data.load_decimated_test_data_set()

ssd_satSys = [SatSys_ssd(sim_data[i].ssd, sim_data[i].name) for i in range(3)]


labels = [r'Nominal urea dosing', r'$+20\%$ urea dosing', r'$-20\%$ urea dosing']
# Plotting response
for i in range(3):
        plt.figure(0)
        plt.plot(ssd_satSys[i].ssd['t'][1:], ssd_satSys[i].ssd['eta'][1:], label=labels[i], color='C'+str(i))
        plt.figure(1)
        plt.plot(ssd_satSys[i].ssd['t'][1:], ssd_satSys[i].eta_hat, label=labels[i])
        plt.fill_between(ssd_satSys[i].ssd['t'][1:],
                         ssd_satSys[i].eta_hat-2*ssd_satSys[i].sigma_eta,
                         ssd_satSys[i].eta_hat+2*ssd_satSys[i].sigma_eta,
                         label=labels[i]+r' $\pm 2\sigma$',
                         color='C'+str(i),
                         alpha=0.2)

plt.figure(0)
plt.grid()
plt.legend()
plt.ylabel(r'$\eta$ '+uc.units['eta'])
plt.xlabel("Time [s]")
# plt.title(r'RMC Test $NO_x$ Reduction with $\pm 20$ Urea Dosing Gain Variation')
plt.savefig("./SatDetection/figs/eta_sim.eps")
plt.figure(1)
plt.grid()
plt.legend()
plt.ylabel(r'$\eta_{sat}$ '+uc.units['eta'])
plt.xlabel("Time [s]")
# plt.title(r'Predicted $NO_x$ Reducion Under Catalyst Saturation')
plt.savefig("./SatDetection/figs/eta_sat.eps")

plt.show()
