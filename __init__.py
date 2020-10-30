#!python3
# __init__.py

# Corban Swain , 2020

import time
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import peak_widths, find_peaks
from matplotlib.gridspec import GridSpec
import c_swain_python_utils as csutils


def gen_info_plot(x, y, wavelength, power):
    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    y_norm = y / peak_val

    width_result = peak_widths(y_norm, [peak_idx])
    peak_width_height = width_result[1][0]
    peak_left = np.interp(width_result[2], np.arange(x.size), x)[0]
    peak_right = np.interp(width_result[3], np.arange(x.size), x)[0]
    peak_width = abs(peak_right - peak_left)
    peak_wavelength = np.average([peak_left, peak_right])
    peak_delta = peak_wavelength - wavelength

    window_size = 50
    upper_lim = wavelength + window_size / 2
    lower_lim = wavelength - window_size / 2

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(position=(0.03, 0.1, 0.57, 0.85))
    ax.plot(x, y_norm, lw=2.5, color='C3')
    ax.set_ylim([-0.02, 1.02])
    ax_lw = 1.5
    sns.despine(ax=ax,
                top=True, right=True, left=True, bottom=False,
                offset=ax_lw * 0)
    ax.spines['bottom'].set_linewidth(ax_lw)
    ax.set_yticks([])
    ax.set_xticks(np.arange(650, 1100, 10))
    ax.set_xticks(np.arange(650, 110, 2), minor=True)
    ax.set_xlim([lower_lim, upper_lim])
    ax.minorticks_on()
    ax.grid(True, 'major', 'x', lw=ax_lw)
    ax.grid(True, 'minor', 'x', lw=ax_lw / 2, ls=':')
    ax.tick_params(axis='x',
                   bottom=True,
                   direction='out',
                   width=ax_lw,
                   length=8)
    ax.annotate('', (peak_wavelength, 1.02), (peak_wavelength, 1.02 + 1e-3),
                arrowprops=dict(headwidth=7,
                                headlength=4,
                                lw=ax_lw,
                                color='C3'),
                annotation_clip=False)
    ax.annotate('', (peak_right, 0.5), (peak_right + window_size / 15, 0.5),
                arrowprops=dict(arrowstyle='->', lw=ax_lw))
    ax.annotate(f'{peak_width:.1f} nm',
                (peak_left, 0.5), (peak_left - window_size / 15, 0.5),
                arrowprops=dict(arrowstyle='->', lw=ax_lw),
                va='center',
                ha='right',
                backgroundcolor='w')

    ax.annotate(f'Expected Wavelength: {wavelength:6.1f} nm\n'
                f'Peak Wavelength:     {peak_wavelength:6.1f} nm\n'
                f'Peak Delta:          {peak_delta:6.1f} nm\n'
                f'Peak FWHM:           {peak_width:6.1f} nm\n'
                f'Measured Max Power:  {power:6.1f} mW',
                xy=(0.95, 0.95),
                xycoords='figure fraction',
                fontfamily='Input',
                va='top',
                ha='right')

    return peak_delta, peak_width



def main():
    data_dir = 'data'
    specta_filename = 'spectra_set_combined.csv'
    power_filename = '201028_power_through_fiber.csv'
    power_norm_filename = '201028_power_at laser.csv'

    spectra_df = pd.read_csv(os.path.join(data_dir, specta_filename),
                             index_col=0)

    power_df = pd.read_csv(os.path.join(data_dir, power_filename),
                           index_col=0,
                           header=None,
                           names=['wavelength', 'power']) * 1e3

    power_norm_df = pd.read_csv(os.path.join(data_dir, power_norm_filename),
                                index_col=0,
                                header=None,
                                names=['wavelength', 'power']) * 1e3

    correction_df = (power_norm_df / power_df).dropna()
    correction_x = correction_df.index.values
    correction_y = correction_df.values.flatten()

    power_x = power_df.index.values

    correction_y_full = np.interp(power_x, correction_x, correction_y)
    power = power_df.values.flatten() * correction_y_full
    power_dict = dict(zip(power_x, power))

    spectra_x = spectra_df.index.values
    spectra_y_names = spectra_df.columns.values.astype(int)
    spectra_y = spectra_df.values

    batches = []
    for i in range(spectra_y_names.size):
        batches.append(dict(x=spectra_x,
                            y=spectra_y[:, i],
                            wavelength=spectra_y_names[i],
                            power=power_dict[spectra_y_names[i]]))

    fig = plt.figure(num=0, figsize=(7, 4))

    results = [gen_info_plot(**b) for b in batches]
    peak_deltas, _peak_widths = tuple(np.array(x) for x in zip(*results))


    gs = GridSpec(3, 1, figure=fig,
                  bottom=0.15,
                  top=0.965,
                  right=0.95,
                  left=0.1,
                  hspace=0.26)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(spectra_y_names, peak_deltas, color='black', lw=1.5)
    ax1.fill_between(spectra_y_names, peak_deltas, 0, where=peak_deltas >= 0,
                     fc='C0', interpolate=True, alpha=0.7)
    ax1.fill_between(spectra_y_names, peak_deltas, 0, where=peak_deltas <= 0,
                     fc='C3', interpolate=True, alpha=0.7)
    ax1.set_yticks([-10, 0])
    ax1.set_ylim([-12, 2])
    ax1.set_ylabel(r'$\Delta\lambda$, nm')

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(spectra_y_names, _peak_widths, color='black', lw=1.5)
    ax2.fill_between(spectra_y_names, _peak_widths, 0, where=_peak_widths >= 0,
                     fc='black', interpolate=True, alpha=0.2)
    ax2.set_yticks([0, 6, 12])
    ax2.set_ylim([0, 12])
    ax2.set_ylabel('FWHM, nm')

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(power_x, power / 1000, color='black', lw=1.5)
    ax3.fill_between(power_x, power / 1000, 0, where=power >= 0,
                     fc='black', interpolate=True, alpha=0.2)
    ax3.set_yticks([0, 0.8, 1.6])
    ax3.set_ylim([0, 1.6])
    ax3.set_xlabel('Expected $\lambda$, nm')
    ax3.set_ylabel('Max Power, W')
    ax_lw = 1

    axs = [ax1, ax2, ax3]

    for i, ax in enumerate(axs):
        sns.despine(ax=ax,
                    top=True, right=True, left=False, bottom=False,
                    offset=ax_lw * 0)
        ax.spines['bottom'].set_linewidth(ax_lw)
        ax.set_xticks(np.arange(690, 1045, 50))
        ax.set_xticks(np.arange(690, 1045, 10), minor=True)
        ax.minorticks_on()
        ax.grid(True, 'major', 'both', lw=ax_lw, zorder=0)
        ax.grid(True, 'minor', 'both', lw=ax_lw / 2, ls=':', zorder=0)
        ax.tick_params(axis='x',
                       bottom=True,
                       direction='out',
                       width=ax_lw,
                       length=5,
                       labelbottom=i == 2)

        ax.set_xlim([690, 1040.1])

    fig.align_ylabels(axs)

    csutils.save_figures('201028_mai_tia_laser_measurements_cswain',
                         add_timestamp=False)

    plt.show()


if __name__ == '__main__':
    main()