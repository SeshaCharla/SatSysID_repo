import numpy as np
from DataProcessing.TestCellData import decimate_data as dd
from DataProcessing.TestCellData import unit_convs as uc
import matplotlib.pyplot as plt
import DataProcessing.TestCellData.sosFiltering as sos
from scipy.signal import sosfiltfilt
from scipy.optimize import lsq_linear, linprog


dg_rmc = dd.decimatedTestData(0, 2)
ag_rmc = dd.decimatedTestData(1, 2)

dg_phi = np.zeros([dg_rmc.ssd_data_len-1, 3])
for i in range(dg_rmc.ssd_data_len-1):
        u2_i = dg_rmc.ssd['u2'][i]
        F_i = dg_rmc.ssd["F"][i]
        T_i = dg_rmc.ssd["T"][i]
        dg_phi[i,:] = (u2_i/F_i**2) * np.array([T_i**2, T_i, 1])

ag_phi = np.zeros([ag_rmc.ssd_data_len-1, 3])
for i in range(ag_rmc.ssd_data_len-1):
        u2_i = ag_rmc.ssd['u2'][i]
        F_i = ag_rmc.ssd["F"][i]
        T_i = ag_rmc.ssd["T"][i]
        ag_phi[i,:] = (u2_i/F_i**2) * np.array([T_i**2, T_i, 1])


dg_sol = linprog(np.sum(dg_phi, axis=0), A_ub=-dg_phi, b_ub = -dg_rmc.ssd['eta'][1:])
ag_sol = linprog(np.sum(ag_phi, axis=0), A_ub=-ag_phi, b_ub = -ag_rmc.ssd['eta'][1:])

dg_theta = dg_sol.x
ag_theta = ag_sol.x
