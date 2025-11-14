import numpy as np
from DataProcessing.TruckData import rdRawDat as rd
from DataProcessing.TruckData import etaCalc
import scipy.signal as sig

drive_cycle_filt = sig.cheby2(7,40,  0.1, 'lowpass', analog=False, fs=1, output='sos')

# Array Manipulating functions ------------------------------------------------------
#==============================================================================================

def find_drive_cycles(t, gap=600):      # 10 min gap
        """Find the discontinuities in the time Data
        The slices would be: [ [t_skips[0], t_skips[1]], [t_skips[1], t_skips[2]],... [t_skips[n-1], t_skips[n]] ]
        """
        t_skips = np.array([i for i in range(1, len(t)) if t[i] - t[i - 1] > gap], dtype=int)
        t_skips = np.append(t_skips, len(t))  # Included the len(t) to follow the slicing rule of open interval
        t_skips = np.insert(t_skips, 0, 0)
        return t_skips

# =============================================================================================

def rmNaNrows(x):
        """Remove the rows with NaN values"""
        return np.delete(x, [i for i in range(len(x)) if np.any(np.isnan(x[i]))], axis=0)

# =============================================================================================

def rmLowTemprows(x):
        """Remove the rows with temperature less than T0.
        The commercial NOx sensor does not work bellow this temperature.
        """
        Tmin = 0   # 200 deg-C
        Tmax = 10   # 300 deg-C
        return np.delete(x, [i for i in range(len(x)) if (x[i, 4]<Tmin or x[i, 4]>Tmax)], axis=0)

#===============================================================================================

class DriveCycle():
        """ Class dividing the truck data into individual drive cycles and linearly interpolating missing data """
        #===========================================================================================
        def __init__(self, age: int, test_type: int):
                self.rawData = rd.RawTruckData(age, test_type)
                self.dt = self.rawData.dt
                self.name = self.rawData.name
                self.iod = self.gen_iod()
                self.drive_cycles = self.gen_drive_cycles()

        # ===================================================

        def get_drive_cycle_data(self, j):
                """ returns the iod data for the ith drive cycle """
                ssd = dict()
                i_min = self.iod['drive_cycles'][j]
                i_max = self.iod["drive_cycles"][j+1]
                ssd['t'] = self.iod['t'][i_min:i_max]
                ssd['y1'] = self.iod["y1"][i_min:i_max]
                ssd['u1'] = self.iod['u1'][i_min:i_max]
                ssd['u2'] = self.iod['u2'][i_min:i_max]



        def gen_drive_cycles(self):
                """ Returns the drive cycles dictionary with all the contiguous filtered data """
                N = len(self.iod['drive_cycles'])-1
                drive_cycles = dict()
                for j in range(N):
                        t = self.iod['t'][self.iod['drive_cycles'][i]:self.iod["drive_cycles"][i+1]]
                        ssd = dict()
                        ssd['t'] = np.arange(np.min(self.iod['t']), np.max(self.iod['t']))
                        ssd['y1'] = np.interp


        # ===========================================================================================

        def gen_iod(self) -> dict[str, np.ndarray]:
                # Generate the input output Data
                raw_tab = np.matrix([self.rawData.raw['t'],
                                     self.rawData.raw['y1'],
                                     self.rawData.raw['u1'],
                                     self.rawData.raw['u2'],
                                     self.rawData.raw['T'],
                                     self.rawData.raw['F']]).T
                iod_tab = rmLowTemprows(rmNaNrows(raw_tab))
                iod_mat = iod_tab.T
                iod = {}
                iod['t'] = np.array(iod_mat[0]).flatten()
                iod['y1'] = np.array(iod_mat[1]).flatten()
                iod['u1'] = np.array(iod_mat[2]).flatten()
                iod['u2'] = np.array(iod_mat[3]).flatten()
                iod['T'] = np.array(iod_mat[4]).flatten()
                iod['F'] = np.array(iod_mat[5]).flatten()
                # Find the time discontinuities in IOD Data
                iod['drive_cycles'] = find_drive_cycles(iod['t'])
                # Set datum for the data
                iod = self.set_datum(iod, type='iod')
                return iod

        #========================================================

        def set_datum(self, ssd):
                """Set the minimum values in data sets"""
                datum = {}
                datum['u1'] = 0.2     # Most of the testcell data shows this
                datum['u2'] = 0
                datum['F'] = 3     # From all the test cell data
                datum['y1'] = 0
                keys = ['y1', 'u1', 'u2', 'F']
                for key in keys:
                        ssd[key] = np.array([val if val >= datum[key] else datum[key] for val in ssd[key]])
                return ssd
