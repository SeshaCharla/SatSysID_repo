import numpy as np
from scipy.io import loadmat
import pathlib as pth
import pickle as pkl
from DataProcessing.TruckData import unit_convs as uc


class RawTruckData():
    """ Class reads the raw-truck data and does the unit-conversions """
    def __init__(self, age: int, trk: int):
        """ Reads the truck data and stores in a .raw dictionary"""
        self.dt = 1
        self .name = self.truck_name(age, trk)
        self.dat_file = self.data_dire()
        try:
            self.raw = self.load_pickle()
        except FileNotFoundError:
            self.raw = self.load_truck_data()
            self.pickle_data()
    # ======================================================================

    def truck_name(self, age: int, trk: int) -> str:
        """ Data names for the truck data
            [0][0-4] - Degreened data
            [1][0-3] - Aged data
        """
        truck = [["adt_15", "mes_15", "wer_15", "trw_15"],
                 ["adt_17", #"adt_17_1", "adt_17_2", "adt_17_3",
                  "mes_18", #"mes_18_1", "mes_18_2", "mes_18_3", "mes_18_4",
                  # "mes_18_5", "mes_18_6", "mes_18_7", "mes_18_8", "mes_18_9",
                  "wer_17", # "wer_17_1", "wer_17_2", "wer_17_3",
                  "trw_16", # "trw_16_1", "trw_16_2", "trw_16_3",
                  # "jnr_16_1", "jnr_16_2", "jnr_16_3", "jnr_16_4", "jnr_16_5",
                  # "wal_16_1", "wal_16_2", "wal_16_3", "wal_16_4", "wal_16_5"
                ]]
        return truck[age][trk]
    # ======================================================================

    def data_dire(self) -> str:
        """ Returns the data directory for the truck data """
        dir_prefix = "./Data"
        truck_dir_prefix = "/drive_data/"
        prefix = dir_prefix + truck_dir_prefix
        truck_dict = {"adt_15": "ADTransport_150814/ADTransport_150814_Day_File.mat",
                      "adt_17": "ADTransport_170201/ADTransport_170201_dat_file.mat",
                      "adt_17_1": "ADTransport_2017_3Days/F_ADTransport_8434_3345_ALA_170131.mat",
                      "adt_17_2": "ADTransport_2017_3Days/F_ADTransport_8434_3345_ALA_170201.mat",
                      "adt_17_3": "ADTransport_2017_3Days/F_ADTransport_8434_3345_ALA_170202.mat",
                      "mes_15": "MesillaValley_150605/MesillaValley_150605_day_file.mat",
                      "mes_18": "MesillaValley_180314/MesillaValley_180314_day_file.mat",
                      "mes_18_1": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180313 (1).mat",
                      "mes_18_2": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180314 (1).mat",
                      "mes_18_3": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180315 (1).mat",
                      "mes_18_4": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180316.mat",
                      "mes_18_5": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180317.mat",
                      "mes_18_6": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180318.mat",
                      "mes_18_7": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180319.mat",
                      "mes_18_8": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180320.mat",
                      "mes_18_9": "MesillaValley_2018_3Days/F_Mesilla_7745_2719_ALA_180321.mat",
                      "wer_15": "Werner_151111/Werner_151111_day_file.mat",
                      "wer_17": "Werner_20171006/Werner_20171006_day_file.mat",
                      "wer_17_1": "Werner_2017_3days/F_Werner_7947_6035_ALA_171005 (1).mat",
                      "wer_17_2": "Werner_2017_3days/F_Werner_7947_6035_ALA_171006 (1).mat",
                      "wer_17_3": "Werner_2017_3days/F_Werner_7947_6035_ALA_171007.mat",
                      "trw_15": "Transwest_150325/Transwest_150325_day_file.mat",
                      "trw_16": "Transwest_161210/Transwest_161210_day_file.mat",
                      "trw_16_1": "Transwest_2016_3Days/F_Transwest_7330_5473_ALA_161208.mat",
                      "trw_16_2": "Transwest_2016_3Days/F_Transwest_7330_5473_ALA_161210 (1).mat",
                      "trw_16_3": "Transwest_2016_3Days/F_Transwest_7330_5473_ALA_161211 (1).mat",
                      "jnr_16_1": "JnRSchugel/F_JnRSchugel_0077_3344_ALA_161101.mat",
                      "jnr_16_2": "JnRSchugel/F_JnRSchugel_0077_3344_ALA_161103.mat",
                      "jnr_16_3": "JnRSchugel/F_JnRSchugel_0077_3344_ALA_161105.mat",
                      "jnr_16_4": "JnRSchugel/F_JnRSchugel_0077_3344_ALA_161106.mat",
                      "jnr_16_5": "JnRSchugel/F_JnRSchugel_0077_3344_ALA_161115.mat",
                      "wal_16_1": "Walmart/F_Walmart_1919_1586_ALA_160124.mat",
                      "wal_16_2": "Walmart/F_Walmart_1919_1586_ALA_160125.mat",
                      "wal_16_3": "Walmart/F_Walmart_1919_1586_ALA_160129.mat",
                      "wal_16_4": "Walmart/F_Walmart_1919_1586_ALA_160130.mat",
                      "wal_16_5": "Walmart/F_Walmart_1919_1586_ALA_160131.mat",
                      }
        return prefix + truck_dict[self.name]
    # =====================================================================================

    def load_truck_data(self):
        # Load the truck Data
        data = loadmat(self.dat_file)
        raw = dict()
        # Assigning the Data to the variables
        Tscr = np.array(data['pSCRBedTemp']).flatten() # 'V_ATP_TRC_SCR_T1'
        raw['t'] = np.array(data['tod']).flatten()
        raw['F'] = uc.uConv(np.array(data['pExhMF']).flatten(), Tscr=Tscr, conv_type="g/s to [x 10 g/s]")                                     # g/sec
        raw['T'] = uc.uConv(Tscr, Tscr=Tscr, conv_type="deg-C to [x 10 + 200 deg C]")
        raw['u2'] = uc.uConv(np.array(data['pUreaDosing']).flatten(), Tscr=Tscr, conv_type="ml/s to [x 10^-1 ml/s]")
        raw['u1'] = uc.uConv(np.array(data['pNOxInppm']).flatten(), Tscr=Tscr, conv_type="ppm to [x 10^-3 mol/m^3]")
        raw['y1'] = uc.uConv(np.array(data['pNOxOutppm']).flatten(),Tscr=Tscr, conv_type="ppm to [x 10^-3 mol/m^3]")
        if self.name == "mes_18":
            for key in raw.keys():
                raw[key] = raw[key][248:]
        return raw
    # ====================================================================================================

    def pickle_data(self):
        # Create a dictionary of the Data
        # Pickle the data_dict to files
        pkl_file = pth.Path("./DataProcessing/TruckData/pkl_files/" + self.name + ".pkl")
        pkl_file.parent.mkdir(parents=True, exist_ok=True)
        with pkl_file.open("wb") as f:
            pkl.dump(self.raw, f)
    # ===============================================================

    def load_pickle(self):
        # Load the pickled Data
        pkl_file = pth.Path("./DataProcessing/TruckData/pkl_files/" + self.name + ".pkl")
        with pkl_file.open("rb") as f:
            raw = pkl.load(f)
        return raw
    # =================================================================

# ======================================================================================================================

def load_truck_data_set():
    """ Loads the entire truck data set """
    ag_trk = [4, 4]
    truck_data = [[RawTruckData(age, trk) for trk in range(ag_trk[age])] for age in range(2)]
    return truck_data

# ======================================================================================================================

## Test

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('qtAgg')

    trk_data = load_truck_data_set()
    ag_trk = [4, 4]

    # Plotting all the data sets
    for age in range(2):
        for trk in range(ag_trk[age]):
            for key in ['u1', 'u2', 'T', 'F', 'y1']:
                plt.figure()
                plt.plot(trk_data[age][trk].raw['t'], trk_data[age][trk].raw[key], label=trk_data[age][trk].name + " " + key, linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel("Time [s]")
                plt.ylabel(key + uc.units[key])
                plt.savefig("./DataProcessing/TruckData/figs/"+trk_data[age][trk].name+"_"+key+".png")
                plt.close()
