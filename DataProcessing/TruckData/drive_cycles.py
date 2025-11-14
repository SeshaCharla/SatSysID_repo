import numpy as np
from DataProcessing.TruckData import rdRawDat as rd
from DataProcessing.TruckData import etaCalc
import scipy.signal as sig
from DataProcessing.TruckData.sosFiltering import drive_cycle_filt


# Array Manipulating functions ------------------------------------------------------
#==============================================================================================

def find_drive_cycles(t, gap=60):      # 10 min gap
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
        def __init__(self, age: int, test_type: int, gap=60):
                self.rawData = rd.RawTruckData(age, test_type)
                self.dt = 1
                self.name = self.rawData.name
                self.iod = self.gen_iod()
                self.drive_cycles = self.gen_drive_cycles()
                self.gap = gap

        # ===================================================

        def sort_drive_cycles(self, drive_cycles_unsorted):
                """ Sorts the drive cycles to get the longest one at 0 """
                drive_cycle_lens = np.array([drive_cycles_unsorted[str(i)]['data_len'] for i in range(self.N_dc)])
                sorted_indices = np.argsort(drive_cycle_lens)
                drive_cycles = dict()
                for i in range(self.N_dc):
                        drive_cycles[str(i)] = drive_cycles_unsorted[str(sorted_indices[-i-1])]
                return drive_cycles


        # ====================================================================

        def get_drive_cycle_data(self, j):
                """ returns the iod data for the ith drive cycle """
                ssd = dict()
                i_min = self.iod['drive_cycles'][j]
                i_max = self.iod["drive_cycles"][j+1]
                for key in ['t', 'y1', 'u1', 'u2', 'T', 'F']:
                        ssd[key] = self.iod[key][i_min:i_max]
                return ssd

        # =======================================================================

        def gen_drive_cycles(self):
                """ Returns the drive cycles dictionary with all the contiguous filtered data """
                self.N_dc = len(self.iod['drive_cycles'])-1
                drive_cycles_unsorted = dict()
                for j in range(self.N_dc):
                        data = self.get_drive_cycle_data(j)
                        ssd = dict()
                        ssd['t'] = np.arange(np.min(data['t']), np.max(data['t']))
                        ssd['data_len'] = len(ssd['t'])
                        for key in ['y1', 'u1', 'u2', 'T', 'F']:
                                interp_data = np.interp(ssd['t'], data['t'], data[key])
                                ssd[key] = sig.sosfiltfilt(drive_cycle_filt, interp_data)
                        ssd = set_datum(ssd)
                        ssd['eta'] = etaCalc.calc_eta(ssd['y1'], ssd['u1'])
                        drive_cycles_unsorted[str(j)] = ssd
                return self.sort_drive_cycles(drive_cycles_unsorted)

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
                iod['drive_cycles'] = find_drive_cycles(iod['t'], gap=self.gap)
                # Set datum for the data
                iod = set_datum(iod)
                return iod

# =======================================================================================================

def set_datum(ssd):
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

# ======================================================================================================================

if __name__ == "__main__":
        import matplotlib.pyplot as plt
        from DataProcessing.TruckData import filt_data
        from DataProcessing.TruckData import plotting as pt

        mes_15 = DriveCycle(1, 2, gap=60)
        mes_15_filt = filt_data.FilteredTruckData(1, 2)

        plt.figure()
        pt.plot_TD(mes_15_filt.iod['t'], mes_15_filt.iod['u1'])
        [plt.plot(mes_15.drive_cycles[str(j)]['t'], mes_15.drive_cycles[str(j)]['u1'], label="drive_cycle_"+str(j)) for j in range(mes_15.N_dc)]
        plt.legend()
        plt.grid()
        plt.show()
