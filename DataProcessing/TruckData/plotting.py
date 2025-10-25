import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkAgg')

def plot_TD(gca, x, y, t_skips, label='none', line_style='-', line_color="tab:orange"):
    """Plots with time skips"""
    for i in range(len(t_skips)-1):
        if i == 0:
            gca.plot(x[t_skips[i]:t_skips[i+1]], y[t_skips[i]:t_skips[i+1]], line_style, label=label, linewidth=1, color=line_color)
        else:
            gca.plot(x[t_skips[i]:t_skips[i + 1]], y[t_skips[i]:t_skips[i + 1]], line_style, linewidth=1,
                     color=line_color)

#======================
tab_lines = iter(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:pink", "tab:olive", "tab:cyan", "tab:purple", "tab:brown"])