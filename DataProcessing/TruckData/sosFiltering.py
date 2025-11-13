import numpy as np
import  scipy.signal as sig

fs = 1
lp_filt = sig.cheby2(7,40,  0.15, 'lowpass', analog=False, fs=1, output='sos')

def sosff_TD(tskips, x: np.ndarray) -> np.ndarray:
    """Filter the data with time jumps"""
    pad_len = 24
    y = np.zeros(np.shape(x))
    for i in range(1, len(tskips)):
        if len(x[tskips[i-1]:tskips[i]]) > pad_len:
            y[tskips[i-1]:tskips[i]] = sig.sosfiltfilt(lp_filt, x[tskips[i-1]:tskips[i]])
        else:
            y[tskips[i - 1]:tskips[i]] = x[tskips[i - 1]:tskips[i]]
    return y

# ==============================================================================================

drive_cycle_filt = sig.cheby2(7,40,  0.1, 'lowpass', analog=False, fs=1, output='sos')

# ===========================================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('qtAgg')

    w, h = sig.sosfreqz(lp_filt, worN=1500)
    wd, hd = sig.sosfreqz(drive_cycle_filt, worN=1500)

    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    db_d = 20*np.log10(np.maximum(np.abs(hd), 1e-5))
    plt.plot((fs/2)*(w/np.pi), db, label="TD filter")
    plt.plot((fs/2)*(wd/np.pi), db_d, label="drive filter")
    plt.ylim(-75, 5)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot((fs/2)*(w/np.pi), np.angle(h), label="TD filter")
    plt.plot((fs/2)*(wd/np.pi), np.angle(hd), label="drive filter")
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency (Nyquist = {})'.format(fs/2))
    plt.legend()
    plt.show()
