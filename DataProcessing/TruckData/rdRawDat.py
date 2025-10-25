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
                 ["adt_17", "mes_18", "wer_17", "trw_16"]]
        return truck[age][trk]
    # ======================================================================

    def data_dire(self) -> str:
        """ Returns the data directory for the truck data """
        dir_prefix = "./Data"
        truck_dir_prefix = "/drive_data/"
        prefix = dir_prefix + truck_dir_prefix
        truck_dict = {"adt_15": "ADTransport_150814/ADTransport_150814_Day_File.mat",
                      "adt_17": "ADTransport_170201/ADTransport_170201_dat_file.mat",
                      "mes_15": "MesillaValley_150605/MesillaValley_150605_day_file.mat",
                      "mes_18": "MesillaValley_180314/MesillaValley_180314_day_file.mat",
                      "wer_15": "Werner_151111/Werner_151111_day_file.mat",
                      "wer_17": "Werner_20171006/Werner_20171006_day_file.mat",
                      "trw_15": "Transwest_150325/Transwest_150325_day_file.mat",
                      "trw_16": "Transwest_161210/Transwest_161210_day_file.mat"}
        return prefix + truck_dict[self.name]
    # =====================================================================================

    def load_truck_data(self):
        # Load the truck Data
        data = loadmat(self.dat_file)
        raw = dict()
        # Assigning the Data to the variables
        Tscr = np.array(data['pSCRBedTemp']).flatten()
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
    truck_data = [[RawTruckData(age, trk) for trk in range(4)] for age in range(2)]
    return truck_data

# ======================================================================================================================

## Test

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('qtAgg')

    trk_data = load_truck_data_set()

    # Plotting all the data sets
    for age in range(2):
        for trk in range(4):
            for key in ['u1', 'u2', 'T', 'F', 'y1']:
                plt.figure()
                plt.plot(trk_data[age][trk].raw['t'], trk_data[age][trk].raw[key], label=trk_data[age][trk].name + " " + key, linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel("Time [s]")
                plt.ylabel(key + uc.units[key])
                plt.savefig("./DataProcessing/TruckData/figs/"+trk_data[age][trk].name+"_"+key+".png")
                plt.close()
