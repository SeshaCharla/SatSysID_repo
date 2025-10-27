import numpy as np
from DataProcessing.SimData import rdRawDat as rd
from DataProcessing.SimData import sosFiltering as sf
from DataProcessing.SimData import etaCalc

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


#===============================================================================================
class FilteredSimData():
    """Class of filtered test data both ssd and iod"""
    #===========================================================================================
    def __init__(self, sim_type: int):
        self.rawData = rd.RawSimData(sim_type)
        self.dt = self.rawData.dt
        self.name = self.rawData.name
        self.ssd = self.gen_ssd()

    # ==========================================================================================
    def gen_ssd(self) -> dict[str, np.ndarray]:
        # Generate the state space Data
        raw_tab = np.matrix([self.rawData.raw['t'],
                             self.rawData.raw['x1'],
                             self.rawData.raw['x2'],
                             self.rawData.raw['u1'],
                             self.rawData.raw['u2'],
                             self.rawData.raw['T'],
                             self.rawData.raw['F'],
                             self.rawData.raw['gamma']]).T
        ssd_tab = rmNaNrows(raw_tab)
        ssd_mat = ssd_tab.T
        ssd = {}
        ssd['t'] = np.array(ssd_mat[0]).flatten()
        ssd['x1'] = np.array(ssd_mat[1]).flatten()
        ssd['x2'] = np.array(ssd_mat[2]).flatten()
        ssd['u1'] = np.array(ssd_mat[3]).flatten()
        ssd['u2'] = np.array(ssd_mat[4]).flatten()
        ssd['T'] = np.array(ssd_mat[5]).flatten()
        ssd['F'] = np.array(ssd_mat[6]).flatten()
        ssd['gamma'] = np.array(ssd_mat[7]).flatten()
        # Find the time discontinuities in SSD Data
        ssd['t_skips'] = find_discontinuities(ssd['t'], self.dt)
        # Smooth all the data
        for state in ['x1', 'x2', 'u1', 'u2', 'T', 'F', 'gamma']:
            ssd[state] = sf.sosff_TD(ssd['t_skips'], ssd[state])
        # Set datum for the data
        ssd = self.set_datum(ssd, type='ssd')
        # Calculating eta
        ssd['eta'] = etaCalc.calc_eta_TD(ssd['x1'], ssd['u1'], ssd['t_skips'])
        return  ssd

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
        datum['gamma'] = 0
        ssd_keys = ['x1', 'x2', 'u1', 'u2', 'F', 'gamma']
        if type == 'ssd':
            key_set = ssd_keys
        else:
            raise ValueError("type must be 'ssd' or 'iod'")
        for key in key_set:
            ssd[key] = np.array([val if val >= datum[key] else datum[key] for val in ssd[key]])
        return ssd


# ======================================================================================================================

## =====================================================================================================================
def load_filtered_sim_data_set():
    # Load the test Data
    ag_tsts = [12, 15]
    filtered_sim_data = [FilteredSimData(sim_type) for sim_type in range(3)]
    return filtered_sim_data

# ======================================================================================================================


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Actually load the entire Data set ----------------------------------------
    sim_data = rd.load_sim_data_set()
    filtered_sim_data = load_filtered_sim_data_set()
    fig_dpi = 300
    keys = ['u1', 'u2', 'T', 'F', 'x1', 'x2', 'eta', 'gamma']

    # Plotting all the Data sets
    for i in range(3):
        for k in range(len(keys)) :
            plt.figure(k)
            key = keys[k]
            plt.plot(filtered_sim_data[i].ssd['t'], filtered_sim_data[i].ssd[key], label= filtered_sim_data[i].name, linewidth=1)
    for k in range(len(keys)):
        plt.figure(k)
        plt.grid()
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel(keys[k])
        plt.title(keys[k])
        plt.savefig("./DataProcessing/SimData/figs/" + "filtered_sim_ssd_" + keys[k] + ".png", dpi=fig_dpi)
    plt.show()

    # Showing datat discontinuities --------------------------------------------
    plt.figure()
    for i in range(3):
        t = filtered_sim_data[i].ssd['t']
        plt.plot(np.arange(len(t)), t, label=sim_data[i].name + 'ss', linewidth=1)
    plt.grid()
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Time [s]')
    plt.title('Time discontinuities in test Data')
    plt.savefig("./DataProcessing/SimData/figs/time_discontinuities_test.png", dpi=fig_dpi)
    plt.close()

    # plt.show()
    plt.close('all')




## Not useful stuff
"""Remove the data from IOD tab where the temperature is less than 200 + y_Tmin deg C"""
"""
# ==================================================================================================================
    def IOD_temp_exclusion(self, iod_tab):
        y_Tmin = 20
        return np.delete(iod_tab,
                         [i for i in range(len(iod_tab))
                                        if (self.rawData.raw['T'])[i]< y_Tmin],
                         axis=0)

"""
