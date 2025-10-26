import numpy as np
from pandas import read_csv
from DataProcessing.SimData import unit_convs as uc


class RawSimData():
    """Class that reads the raw test data
     Does the unit conversions"""

    # ==============================================================
    def __init__(self, sim_type:int):
        """Reads the test data and stores into a .raw dictionary"""
        self.dt = 1           # Sampling time
        sim_names = ["sim_nom", "sim_+20", "sim_-20"]
        self.name = sim_names[sim_type]
        # print(self.name)
        self.dat_file = self.data_dir()
        self.raw = self.load_sim_data()

    # ==============================================================
    def load_sim_data(self) -> dict[str, np.ndarray]:
        """Loads the test data"""
        data = read_csv(self.dat_file, header=[0, 1])
        raw_data = {}
        # ======================================================================================
        # Assigning the Data to the variables
        # Time is in seconds
        raw_data['t'] = np.array(data.get(data.columns[0]), dtype=np.float64).flatten()
        # print("len(t) = ", np.shape(raw_data['t']))
        # ======================================================================================
        # Temperature is in deg-C
        Tin = np.array(data.get(data.columns[1]), dtype=np.float64).flatten()
        Tout = np.array(data.get(data.columns[2]), dtype=np.float64).flatten()
        Tscr = np.mean([Tin, Tout], axis=0).flatten()
        raw_data['T'] = uc.uConv(Tscr, Tscr, "-T0C")
        # print("len(T) = ", np.shape(raw_data['T']))
        # ======================================================================================
        # Mass flow rate is in g/sec
        F_kgmin = np.array(data.get(data.columns[3]), dtype=np.float64).flatten()
        raw_data['F'] = uc.uConv(F_kgmin,Tscr, "kg/min to 10 g/s")        # g/sec
        # print("len(F) = ", np.shape(raw_data['F']))
        # =======================================================================================
        # NOx output is in mol/m^3
        NOx = np.array(data.get(data.columns[6]), dtype=np.float64).flatten()
        raw_data['x1'] = uc.uConv(NOx, Tscr, "ppm to 10^-3 mol/m^3")
        # print("len(x1) = ", np.shape(raw_data['x1']))
        # =======================================================================================
        # NH3 output is in mol/m^3
        NH3 = np.array(data.get(data.columns[7]), dtype=np.float64).flatten()
        raw_data['x2'] = uc.uConv(NH3, Tscr, "ppm to 10^-3 mol/m^3")
        # print("len(x2) = ", np.shape(raw_data['x2']))
        # =======================================================================================
        # NOx input is in mol/m^3
        u1 = np.array(data.get(data.columns[4]), dtype=np.float64).flatten()
        raw_data['u1'] = uc.uConv(u1, Tscr, "ppm to 10^-3 mol/m^3")
        # print("len(u1) = ", np.shape(raw_data['u1']))
        # ========================================================================================
        # Urea injection rate is in ml/sec
        u2 = np.array(data.get(data.columns[5]), dtype=np.float64).flatten()
        raw_data['u2'] = uc.uConv(u2, Tscr, "ml/s to 10^-1 ml/s")
        # print("len(u2) = ", np.shape(raw_data['u2']))
        # u1_sensor = np.array(Data.get(('EONOX_COMP_VALUE', 'ppm'))).flatten()
        # ======================================================================================================
        # Ammonia coverage ratio at the inlet
        gamma = np.array(data.get(data.columns[8]), dtype=np.float64).flatten()
        raw_data['gamma'] = gamma
        return raw_data

    # ==============================================================
    # ==============================================================
    def data_dir(self) -> str:
        """Returns the data directory for the test data"""
        dir_prefix = "./Data"
        sim_dir_prefix = "/sim_data/"
        sim_dict = {"sim_nom": sim_dir_prefix + "Sim_Results_Nominal_DG.csv",
                    "sim_+20": sim_dir_prefix + "Sim_Results_DEF_+20.csv",
                    "sim_-20": sim_dir_prefix + "Sim_Results_DEF_-20.csv"
                     }
        return dir_prefix + sim_dict[self.name]

    # ===============================================================

## =====================================================================================================================
def load_sim_data_set():
    # Load the test Data
    sim_data = [RawSimData(sim) for sim in range(3)]
    return sim_data

# ======================================================================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Actually load the entire Data set ----------------------------------------
    sim_data = load_sim_data_set()
    fig_dpi = 300

    # Plotting all the Data sets
    for i in range(3):
        for key in ['u1', 'u2', 'T', 'F', 'x1', 'x2', 'gamma']:
            plt.figure()
            plt.plot(sim_data[i].raw['t'], sim_data[i].raw[key], label=sim_data[i].name + " " + key, linewidth=1)
            plt.grid()
            plt.legend()
            plt.xlabel('Time [s]')
            plt.ylabel(key)
            plt.title(sim_data[i].name)
            plt.savefig("./DataProcessing/SimData/figs/" + sim_data[i].name + "_raw_" + key + ".png", dpi=fig_dpi)
            plt.close()
