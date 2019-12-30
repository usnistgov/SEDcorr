# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:49:40 2019

This script works up the data that is the basis for Figures 2b, Figure 5, and
Figure 6a-c from the manuscript.  It serves as an example for how to use the
systematic energy deficit correction code.

It requires the numpy, matplotlib, and colorcet packages to be installed.

@author: bwc

>  NIST Public License - 2019

>  This software was developed by employees of the National Institute of
>  Standards and Technology (NIST), an agency of the Federal Government
>  and is being made available as a public service. Pursuant to title 17
>  United States Code Section 105, works of NIST employees are not subject
>  to copyright protection in the United States.  This software may be
>  subject to foreign copyright.  Permission in the United States and in
>  foreign countries, to the extent that NIST may hold copyright, to use,
>  copy, modify, create derivative works, and distribute this software and
>  its documentation without fee is hereby granted on a non-exclusive basis,
>  provided that this notice and disclaimer of warranty appears in all copies.

>  THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND,
>  EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
>  TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
>  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
>  AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION
>  WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE
>  ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING,
>  BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES,
>  ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE,
>  WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER
>  OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND
>  WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF,
>  OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.

"""


import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

from apt_io import read_epos_numpy
from sed_corr import get_all_scale_coeffs
from histo_funcs import edges_to_centers
from time import time

# Load epos file
epos_fn = r'R45_04472-v03.epos'
epos = read_epos_numpy(epos_fn)


def create_histogram(xs, ys, cts_per_slice=2**10, y_roi=None, delta_y=0.1):
    """Create a 2d histogram of the data, specifying the bin intensity, region
    of interest (on the y-axis), and the spacing of the y bins"""
    # even number
    num_y = int(np.ceil(np.abs(np.diff(y_roi))/delta_y/2)*2)
    num_x = int(ys.size/cts_per_slice)
    return np.histogram2d(xs, ys, bins=[num_x, num_y],
                          range=[[0, max(xs)], y_roi],
                          density=False)


def _extents(f):
    """Helper function to determine axis extents based off of the bin edges"""
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


def plot_2d_histo(ax, N, x_edges, y_edges):
    """Helper function to plot a histogram on an axis"""
    ax.imshow(np.log10(1+np.transpose(N)), aspect='auto',
              extent=_extents(x_edges) + _extents(y_edges),
              origin='lower', cmap=cc.cm.CET_L8,
              interpolation='bilinear')


# Perform correction on IVAS voltage and bowl corrected mass-to-charge data

# Note: This differs slightly from the manuscript since the scale corrections
# in the manuscript were computed for the time-of-flight data for clarity.  In
# practice it is easier in most cases to perform the correction on
# the mass-to-charge data as is done here.  There will also be small
# differences due to voltage and bowl corrections that are slightly different
# (relative to the manuscript Figures).
t_start = time()
eventwise_scales = get_all_scale_coeffs(epos['m2q'],
                                        max_scale=1.15,
                                        roi=[0.8, 80],
                                        cts_per_chunk=2**7,
                                        delta_logdat=5e-4)
t_end = time()
print('time for correction: %.3f s' % (t_end-t_start))
print('rate for correction: %.3e ions/s' % (epos.size/(t_end-t_start)))

corrected_m2q = epos['m2q']/eventwise_scales

# Plot the results
fig_num = 101
plt.close(fig_num)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                             figsize=(11, 8.5),
                                             num=fig_num)

event_idx = np.arange(0, epos.size)/1000000

plot_2d_histo(ax1, *create_histogram(event_idx, epos['m2q'], y_roi=[13, 18],
                                     cts_per_slice=2**7, delta_y=0.03))
ax1.set(xlabel='event index (millions)', ylabel='m/z (Da)',
        title='Histogram of Mass to charge (uncorrected)\n'
              'with applied voltage (V$_{\\rm dc}$)',
        xlim=[0, max(event_idx)])

ax1_twin = ax1.twinx()
ax1_twin.plot(event_idx, epos['v_dc'],
              '-',
              markersize=.1,
              marker=',',
              markeredgecolor='#1f77b4aa',
              color='white')
ax1_twin.set(ylabel=r'V$_{\rm dc}$')
leg = ax1_twin.legend([r'V$_{\rm dc}$'], loc='lower right',
                      facecolor='#000e5c', framealpha=1)
leg.get_texts()[0].set_color('white')

plot_2d_histo(ax2, *create_histogram(event_idx, corrected_m2q, y_roi=[13, 18],
                                     cts_per_slice=2**7, delta_y=0.03))
ax2.set(xlabel='event index (millions)', ylabel='m/z (Da)',
        title='Histogram of Mass to charge (corrected)\n'
              'with applied voltage (V$_{\\rm dc}$)',
        xlim=[0, max(event_idx)])

ax2_twin = ax2.twinx()
ax2_twin.plot(event_idx, epos['v_dc'],
              '-',
              markersize=.1,
              marker=',',
              markeredgecolor='#1f77b4aa',
              color='white')
ax2_twin.set(ylabel=r'V$_{\rm dc}$')
ax2_twin.legend([r'V$_{\rm dc}$'], loc='upper right', framealpha=1)
leg = ax2_twin.legend([r'V$_{\rm dc}$'], loc='lower right',
                      facecolor='#000e5c', framealpha=1)
leg.get_texts()[0].set_color('white')

ax3.plot(event_idx, eventwise_scales, lw=0.5)
ax3.set(xlabel='event index (millions)',
        ylabel='correction factor',
        xlim=[0, max(event_idx)],
        title='Systematic Energy Deficit Correction factor')
ax3.grid()


m2q_binsize = 0.03
m2q_roi = [0, 80]
nbins = int((m2q_roi[1]-m2q_roi[0])/m2q_binsize)

before_SED_counts, bin_edges = np.histogram(epos['m2q'],
                                            bins=nbins,
                                            range=m2q_roi)
bin_centers = edges_to_centers(bin_edges)[0]
ax4.fill_between(bin_centers, 0, before_SED_counts, lw=1,
                 label='Before SED correction')

after_SED_counts, bin_edges = np.histogram(corrected_m2q,
                                           bins=nbins,
                                           range=m2q_roi)
bin_centers = edges_to_centers(bin_edges)[0]
ax4.fill_between(bin_centers, before_SED_counts, after_SED_counts * 10,
                 lw=1, label='After SED correction')
ax4.set(xlabel='m/z (Da)', ylabel='counts',
        xlim=[0, 65],
        title='Comparison of uncorrected vs. SED-corrected\nmass spectra '
              '(shifted for clarity)')
ax4.tick_params(axis='y', which='both', left=False, right=False,
                labelleft=False, labelright=False)
ax4.legend()

ax4.set_yscale('log')

fig.tight_layout()
# plt.show()
fig.savefig('example_script_output.png', dpi=300)
