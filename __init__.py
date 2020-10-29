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
from scipy.signal import peak_widths


def gen_info_plot(x, y, wavelength, peak_idx):
    peak_val = y[peak_idx]
    peak_wavelength = x[peak_idx]
    y_norm = y / peak_val

    width_result = peak_widths(y_norm, [peak_idx])
    peak_width_height = width_result[1][0]
    peak_left = np.interp(width_result[2], np.arange(x.size), x)[0]
    peak_right = np.interp(width_result[3], np.arange(x.size), x)[0]
    peak_width = abs(peak_right - peak_left)

    window_size = 50

    upper_lim = wavelength + window_size / 2
    lower_lim = wavelength - window_size / 2

    sns.set_style('whitegrid')
    fig = plt.figure()
    plt.plot(x, y_norm, lw=3)
    plt.ylim([0, 1.05])
    sns.despine(fig, top=True, right=True, left=True, bottom=False, offset=5)
    plt.yticks([])
    plt.xticks(np.arange(650, 1100, 10))
    plt.xlim([lower_lim, upper_lim])
    plt.tick_params(axis='x',
                    bottom=True,
                    direction='out',
                    length=4)

    plt.show()



def main():
    data_dir = 'data'
    specta_filename = 'spectra_set_combined.csv'

    spectra_df = pd.read_csv(os.path.join(data_dir, specta_filename),
                             index_col=0)

    spectra_x = spectra_df.index.values
    spectra_y_names = spectra_df.columns.values.astype(int)
    spectra_y = spectra_df.values

    peak_idx = np.argmax(spectra_y, axis=0)

    batches = []

    for i in range(spectra_y_names.size):
        batches.append(dict(x=spectra_x,
                            y=spectra_y[:, i],
                            wavelength=spectra_y_names[i],
                            peak_idx=peak_idx[i]))

    [gen_info_plot(**b) for b in batches]

if __name__ == '__main__':
    main()