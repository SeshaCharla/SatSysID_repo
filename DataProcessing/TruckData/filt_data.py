import numpy as np
from DataProcessing.TruckData import rdRawDat as rd
from DataProcessing.TruckData import sosFiltering as sf
from DataProcessing.TruckData import etaCalc

# Array Manipulating functions ------------------------------------------------------
#==============================================================================================
def find_discontinuities(t, dt):
    """Find the discontinuities in the time Data
    The slices would be: [ [t_skips[0], t_skips[1]], ... [t_skips[n-1], t_skips[n]] ]
    """
    t_skips = np.array([i for i in range(1, len(t))
                        if t[i] - t[i - 1] > 1.5 * dt], dtype=int)
    t_skips = np.append(t_skips, len(t))            # Included the len(t) to follow the slicing rule of open interval
    t_skips = np.insert(t_skips, 0, 0)
    return t_skips


# =============================================================================================
def rmNaNrows(x):
    """Remove the rows with NaN values"""
    return np.delete(x,
                     [i for i in range(len(x))
                         if np.any(np.isnan(x[i]))],
                     axis=0)

# =============================================================================================
def rmLowTemprows(x):
    """Remove the rows with temperature less than T0.
        The commercial NOx sensor does not work bellow this temperature.
    """
    delta = 1e-5
    Tmin = 0+delta    # 200 deg-C
    Tmax = ((360-200)/10) - delta
    return np.delete(x,
                     [i for i in range(len(x))
                         if (x[i, 4]<Tmin or x[i, 4]>Tmax)],
                     axis=0)

#===============================================================================================
class FilteredTruckData():
    """Class of filtered test data both ssd and iod"""
    #===========================================================================================
    def __init__(self, age: int, test_type: int):
        self.rawData = rd.RawTruckData(age, test_type)
        self.dt = self.rawData.dt
        self.name = self.rawData.name
        self.iod = self.gen_iod()

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
        iod['t_skips'] = find_discontinuities(iod['t'], self.dt)
        # Smooth all the data
        for state in ['y1', 'u1', 'u2', 'T', 'F']:
            iod[state]= sf.sosff_TD(iod['t_skips'], iod[state])
        # Set datum for the data
        iod = self.set_datum(iod, type='iod')
        # Calculate eta
        iod['eta'] = etaCalc.calc_eta_TD(iod['y1'], iod['u1'], iod['t_skips'])
        return iod

    #===================================================================================================================
    def set_datum(self, ssd, type='ssd'):
        """Set the minimum values in data sets"""
        datum = {}
        datum['x1'] = 0
        datum['x2'] = 0
        datum['u1'] = 0.2     # Most of the testcell data shows this
        datum['u2'] = 0
        datum['F'] = 3     # From all the test cell data
        datum['y1'] = 0
        ssd_keys = ['x1', 'x2', 'u1', 'u2', 'F']
        iod_keys = ['y1', 'u1', 'u2', 'F']
        if type == 'ssd':
            key_set = ssd_keys
        elif type == 'iod':
            key_set = iod_keys
        else:
            raise ValueError("type must be 'ssd' or 'iod'")
        for key in key_set:
            ssd[key] = np.array([val if val >= datum[key] else datum[key] for val in ssd[key]])
        return ssd


# ======================================================================================================================

## =====================================================================================================================
def load_filtered_truck_data_set():
    # Load the test Data
    ag_tsts = [4, 4]
    filtered_truck_data = [[FilteredTruckData(age, tst) for tst in range(ag_tsts[age])] for age in range(2)]
    return filtered_truck_data

# ======================================================================================================================


if __name__ == '__main__':
    from plotting import *
    import unit_convs as uc

    # Actually load the entire Data set ----------------------------------------
    test_data = rd.load_truck_data_set()
    filtered_test_data = load_filtered_truck_data_set()
    fig_dpi = 300
    ag_tsts = [4, 4]

    # Plotting all the Data sets
    for i in range(2):
        for j in range(ag_tsts[i]):
            for key in ['u1', 'u2', 'T', 'F', 'y1', 'eta']:
                plt.figure()
                if (key != 'eta'):
                    plt.plot(test_data[i][j].raw['t'], test_data[i][j].raw[key], '--', label=key, linewidth=1, color='tab:blue')
                plot_TD(plt.gca(), filtered_test_data[i][j].iod['t'], filtered_test_data[i][j].iod[key],
                                    filtered_test_data[i][j].iod['t_skips'], label=key + "_filtered")
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key + uc.units[key])
                plt.title(test_data[i][j].name + "_iod")
                plt.savefig("./DataProcessing/TruckData/figs/" + test_data[i][j].name + "_iod_" + key + ".png", dpi=fig_dpi)
                plt.close()

    # Showing datat discontinuities --------------------------------------------
    plt.figure()
    for i in range(2):
        for j in range(3):
            t = filtered_test_data[i][j].iod['t']
            plt.plot(np.arange(len(t)), t, label=test_data[i][j].name + 'io', linewidth=1)
    plt.grid()
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Time [s]')
    plt.title('Time discontinuities in test Data')
    plt.savefig("./DataProcessing/TruckData/figs/"+"time_discontinuities_test.png", dpi=fig_dpi)
    plt.close()

    # plt.show()
    plt.close('all')
