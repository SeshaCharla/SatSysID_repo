import numpy as np
from DataProcessing.SimData import filt_data
from DataProcessing.SimData import unit_convs as uc
from SatDetection import SatParmID as spd
import matplotlib.pyplot as plt
from SatSysID import SatSysID_funcs as sf

# ==============================================

sim_data = filt_data.load_filtered_sim_data_set()
# tst_data = decimate_data.load_decimated_test_data_set()

thetas = np.zeros([3, 3])
for i in range(3):
        Phi = spd.PhiSat_mat(sim_data[i].ssd['T'], sim_data[i].ssd['F'], sim_data[i].ssd['u1'])
        H = np.matrix(sim_data[i].ssd['eta']).T
        thetas[i,:] = sf.solve_QP(Phi[0:-1,:], H[1:, :]).T

labels = [r'Nominal urea dosing', r'$+20\%$ urea dosing', r'$-20\%$ urea dosing']
# Plotting response
for i in range(3):
        plt.figure(0)
        plt.plot(sim_data[i].ssd['t'], sim_data[i].ssd['eta'], label=labels[i])
        plt.figure(1)
        plt.plot(sim_data[i].ssd['t'], spd.pred_sat_response(np.matrix(thetas[i,:]).T, sim_data[i].ssd['T'],
                                                             sim_data[i].ssd['F'], sim_data[i].ssd['u1']) ,
                 label=labels[i])
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
