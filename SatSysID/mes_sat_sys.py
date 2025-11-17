# Saturated system analysis for Messilla Valley truck data
import numpy as np
from DataProcessing.TruckData import drive_cycles as dc
import matplotlib.pyplot as plt
from SatSysID import SatSysID_funcs as sf
from SatSysID.SatSysID_methods import SatSys_ssd

dg_trk = dc.DriveCycle(0, 1, gap=60)
ag_trk = dc.DriveCycle(1, 1, gap=60)

dg_ssd = [dg_trk.drive_cycles[str(j)] for j in range(dg_trk.N_dc)]
ag_ssd = [ag_trk.drive_cycles[str(j)] for j in range(ag_trk.N_dc)]

dg_ssd_satSys = [SatSys_ssd(dg_ssd[i], dg_trk.name + "_"+ str(i)) for i in range(dg_trk.N_dc)]
ag_ssd_satSys = [SatSys_ssd(ag_ssd[i], ag_trk.name + "_"+ str(i)) for i in range(ag_trk.N_dc)]

for i in range(len(dg_ssd_satSys)):
        theta = dg_ssd_satSys[i].theta.copy()
        if (theta[0, 0] < 0 and theta[1, 0] > 0 and theta[2, 0] > 0):
                dg_theta_ref = theta
                break
        else:
                continue
ref = 0
for i in range(dg_trk.N_dc):
        dg_ssd_satSys[i].predict_eta_sat(dg_ssd_satSys[ref].ssd)
        dg_ssd_satSys[i].calc_Tw(dg_theta_ref)
for i in range(ag_trk.N_dc):
        ag_ssd_satSys[i].predict_eta_sat(dg_ssd_satSys[ref].ssd)
        ag_ssd_satSys[i].calc_Tw(dg_theta_ref)

ag_Tw = [sys.Tw for sys in ag_ssd_satSys]
dg_Tw = [sys.Tw for sys in dg_ssd_satSys]

plt.figure()
plt.hist(dg_Tw, label=dg_trk.name, bins=100)
plt.hist(ag_Tw, label=ag_trk.name, bins=100)
plt.legend()
plt.show()
