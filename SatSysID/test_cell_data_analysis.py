import numpy as np
from DataProcessing.TestCellData import decimate_data as dd
from DataProcessing.TestCellData import unit_convs as uc
import matplotlib.pyplot as plt
from SatSysID import SatSysID_funcs as sf
from SatSysID.SatSysID_methods import SatSys_ssd

# =====================================================
j = 2
set_names = ['cftp', 'hftp', 'rmc']
data_set = set_names[j]
dat = [dd.decimatedTestData(0, j+3*i) for i in range(1,4)] + [dd.decimatedTestData(1, j+3*i) for i in range(1)]

N = len(dat)
# tst_data = decimate_data.load_decimated_test_data_set()

ssd_sat_sys = [SatSys_ssd(dat[i].ssd, dat[i].name) for i in range(N)]
iod_sat_sys = [SatSys_ssd(dat[i].iod, dat[i].name) for i in range(N)]
for i in range(N):
        ssd_sat_sys[i].predict_eta_sat(dat[-1].ssd)
        iod_sat_sys[i].predict_eta_sat(dat[-1].iod)



# Plotting response ssd
for i in range(N):
        plt.figure(0)
        plt.plot(dat[i].ssd['t'], dat[i].ssd['eta'], label=dat[i].name, color='C'+str(i))
        plt.figure(1)
        plt.plot(ssd_sat_sys[i].ssd_ref['t'][1:], ssd_sat_sys[i].eta_pred, label=ssd_sat_sys[i].name + r' $\pm \sigma$')
        plt.fill_between(ssd_sat_sys[i].ssd_ref['t'][1:],
                         ssd_sat_sys[i].eta_pred-1*ssd_sat_sys[i].sigma_pred,
                         ssd_sat_sys[i].eta_pred+1*ssd_sat_sys[i].sigma_pred,
                         label=None,
                         color='C'+str(i),
                         alpha=0.2)
        plt.figure(2)
        plt.plot(iod_sat_sys[i].ssd_ref['t'][1:], iod_sat_sys[i].eta_pred, label=iod_sat_sys[i].name + r' $\pm 2 \sigma$')
        plt.fill_between(iod_sat_sys[i].ssd_ref['t'][1:],
                         iod_sat_sys[i].eta_pred-2*iod_sat_sys[i].sigma_pred,
                         iod_sat_sys[i].eta_pred+2*iod_sat_sys[i].sigma_pred,
                         label=None,
                         color='C'+str(i),
                         alpha=0.2)


plt.figure(0)
plt.grid()
plt.legend()
plt.ylabel(r'$\eta$ '+uc.units['eta'])
plt.xlabel("Time [s]")
# plt.title(r'RMC Test $NO_x$ Reduction with $\pm 20$ Urea Dosing Gain Variation')
plt.savefig("./SatDetection/figs/eta_"+"dat_set"+".png", dpi=300)
plt.figure(1)
plt.grid()
plt.legend()
plt.ylabel(r'$\eta_{sat}$ '+uc.units['eta'])
plt.xlabel("Time [s]")
plt.title("FTIR Measurements")
plt.savefig("./SatDetection/figs/eta_sat_"+"dat_set"+".png", dpi=300)
plt.figure(2)
plt.grid()
plt.legend()
plt.ylabel(r'$\eta_{sat}$ '+uc.units['eta'])
plt.xlabel("Time [s]")
plt.title(r'$NO_x$ Sensor Measurements')
plt.savefig("./SatDetection/figs/eta_sat_iod_"+"dat_set"+".png", dpi=300)



plt.show()
