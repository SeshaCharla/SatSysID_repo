import numpy as np
import decimation as dc
import filt_data as fd
import etaCalc
import scipy.signal as sig
from sosFiltering import fs1_filt


class decimatedTestData():
    """ The Class of decimated data """
    # ========================================================================
    def __init__(self, age: int, test_type: int):
        """ Initializes the data classes """
        self.filtData = fd.FilteredTestData(age, test_type)
        self.dt = 1
        self.name = self.filtData.name
        self.ssd = self.decimate_ssd()
        self.iod = self.decimate_iod()
        self.ssd_data_len = len(self.ssd['t'])
        self.iod_data_len = len(self.iod['t'])

    # ========================================================================
    def decimate_ssd(self) -> dict[str, np.ndarray]:
        """ Decimate the filtered ssd data """
        ssd_keys = ['x1', 'x2', 'u1', 'u2', 'F', 'T', 'eta', 'mu']
        ssd = {}
        for key in ssd_keys:
            ssd[key] = dc.decimate_withTD(self.filtData.ssd['t_skips'], self.filtData.ssd[key])
        ssd['t'] = dc.decimate_time2OneHz(self.filtData.ssd['t_skips'], self.filtData.ssd['t'])
        if self.name == "dg_rmc_2":
            for key in ssd_keys:
                ssd[key] = dc.sig.decimate(self.filtData.ssd[key], q=10, n=7, ftype='iir', zero_phase=True)
            ssd['t'] = np.arange(0, self.filtData.ssd['t'][-1]+1, 1)
        ssd['t_skips'] = fd.find_discontinuities(ssd['t'], self.dt)
        ssd['eta_dec'] = ssd['eta']
        ssd['eta'] = etaCalc.calc_eta_TD(ssd['x1'], ssd['u1'], ssd['t_skips'])
        ssd['u1F'] = sig.sosfiltfilt(fs1_filt, np.array([ssd['u1'][i]/ssd['F'][i] for i in range(len(ssd['u1']))]))
        ssd['u2F'] = sig.sosfiltfilt(fs1_filt, np.array([ssd['u2'][i]/ssd['F'][i] for i in range(len(ssd['u2']))]))
        ssd['eta_F'] = sig.sosfiltfilt(fs1_filt, etaCalc.calc_eta_F(ssd['x1'], ssd['u1'], ssd['F']))
        return ssd

    # =========================================================================
    def decimate_iod(self):
        """ Decimate the filtered iod data """
        iod_keys = ['y1', 'u1', 'u2', 'F', 'T', 'eta', 'mu']
        iod = {}
        for key in iod_keys:
            iod[key] = dc.decimate_withTD(self.filtData.iod['t_skips'], self.filtData.iod[key])
        iod['t'] = dc.decimate_time2OneHz(self.filtData.iod['t_skips'], self.filtData.iod['t'])
        if self.name == "dg_rmc_2":
            for key in iod_keys:
                iod[key] = dc.sig.decimate(self.filtData.iod[key], q=10, n=7, ftype='iir', zero_phase=True)
            iod['t'] = np.arange(0, self.filtData.iod['t'][-1]+1, 1)
        iod['t_skips'] = fd.find_discontinuities(iod['t'], self.dt)
        iod['eta_dec'] = iod['eta']
        iod['eta'] = etaCalc.calc_eta_TD(iod['y1'], iod['u1'], iod['t_skips'])
        iod['u1F'] = sig.sosfiltfilt(fs1_filt, np.array([iod['u1'][i]/iod['F'][i] for i in range(len(iod['u1']))]))
        iod['u2F'] = sig.sosfiltfilt(fs1_filt, np.array([iod['u2'][i]/iod['F'][i] for i in range(len(iod['u2']))]))
        iod['eta_F'] = sig.sosfiltfilt(fs1_filt, etaCalc.calc_eta_F(iod['y1'], iod['u1'], iod['F']))
        return iod

# ======================================================================================================================

## =====================================================================================================================
def load_decimated_test_data_set():
    # Load the test Data
    ag_tsts = [12, 15]
    decimated_test_data = [[decimatedTestData(age, tst) for tst in range(ag_tsts[age])] for age in range(2)]
    return decimated_test_data

# ======================================================================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import unit_convs as uc
    mpl.use('tkAgg')

    # Actually load the entire Data set ----------------------------------------
    dct = load_decimated_test_data_set()
    fig_dpi = 300
    ag_tsts = [12, 15]
    show_plot = None

    # Plotting all the Data sets
    for i in range(2):
        for j in range(ag_tsts[i]):
            for key in ['u1', 'u2', 'T', 'F', 'x1', 'x2', 'eta', 'mu', 'u1F', 'u2F', 'eta_F']:
                plt.figure()
                if (key != 'eta' and key != 'u1F' and key != 'u2F' and key != 'eta_F'):
                    plt.plot(dct[i][j].filtData.rawData.raw['t'], dct[i][j].filtData.rawData.raw[key], ':', label=key+"_raw", linewidth=1)
                if (key != 'u1F' and key != 'u2F' and key != 'eta_F'):
                    plt.plot(dct[i][j].filtData.ssd['t'], dct[i][j].filtData.ssd[key], '-.', label= key+"_filtered", linewidth=1)
                plt.plot(dct[i][j].ssd['t'], dct[i][j].ssd[key], '--',label=key + "_decimated", linewidth=1)
                if (key == 'eta'):
                    plt.plot(dct[i][j].ssd['t'], dct[i][j].ssd['eta_dec'], '--', label="decimate(eta_filtered)", linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key + uc.units[key])
                plt.title(dct[i][j].name + "_ssd")
                plt.savefig("./DataProcessing/TestCellData/figs/" + dct[i][j].name + "_ssd_" + key + ".png", dpi=fig_dpi)
                if key != show_plot:
                    plt.close()
                else:
                    plt.show()

            for key in ['u1', 'u2', 'T', 'F', 'y1', 'eta', 'mu', 'u1F', 'u2F', 'eta_F']:
                plt.figure()
                if (key != 'eta' and key != 'u1F' and key != 'u2F' and key != 'eta_F'):
                    plt.plot(dct[i][j].filtData.rawData.raw['t'], dct[i][j].filtData.rawData.raw[key], ':', label=key+"_raw", linewidth=1)
                if (key != 'u1F' and key != 'u2F' and key != 'eta_F'):
                    plt.plot(dct[i][j].filtData.iod['t'], dct[i][j].filtData.iod[key], '-.', label= key+"_filtered", linewidth=1)
                plt.plot(dct[i][j].iod['t'], dct[i][j].iod[key], '--', label=key + "_decimated", linewidth=1)
                if (key == 'eta'):
                    plt.plot(dct[i][j].iod['t'], dct[i][j].iod['eta_dec'], '--', label="decimate(eta_filtered)", linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key + uc.units[key])
                plt.title(dct[i][j].name + "_iod")
                plt.savefig("./DataProcessing/TestCellData/figs/" + dct[i][j].name + "_iod_" + key + ".png", dpi=fig_dpi)
                if key != show_plot:
                    plt.close()
                else:
                    plt.show()

    # Showing datat discontinuities --------------------------------------------
    plt.figure()
    for i in range(2):
        for j in range(3):
            t = dct[i][j].ssd['t']
            plt.plot(np.arange(len(t)), t, '--', label=dct[i][j].name + 'ss', linewidth=1)
            t = dct[i][j].iod['t']
            plt.plot(np.arange(len(t)), t, '--', label=dct[i][j].name + 'io', linewidth=1)
    plt.grid()
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Time [s]')
    plt.title('Time discontinuities in test Data')
    plt.savefig("./DataProcessing/TestCellData/figs/time_discontinuities_test.png", dpi=fig_dpi)
    plt.close()

    # plt.show()
    plt.close('all')
