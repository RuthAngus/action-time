"""
MONTE_CARLO_ACTIONS
===================
This code uses the covariances on RA, dec, pmra and pmdec to calculate actions
with uncertainties.
It also uses the Bailer-Jones posterior for distances.
"""

import numpy as np
import matplotlib.pyplot as plt
# import paia
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import corner

plotpar = {'axes.labelsize':  18,
           'font.size':       18,
           'legend.fontsize': 18,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def gen_sample_set(d, N, parallax=False, plot=False):
    """
    Generate all the samples needed for this analysis by sampling from the
    (Gaussian assumed) likelihood functions.
    params
    ======
    d: (pandas DataFrame)
       dataframe containing Gaia data for one star.
    N: (int)
        Number of samples.
    """

    mus = np.array([d.ra.values, d.dec.values, d.pmra.values,
                    d.pmdec.values])
    C = np.matrix([
        [d.ra_error.values**2, d.ra_dec_corr.values,
            d.ra_pmra_corr.values, d.ra_pmdec_corr.values],
        [d.ra_dec_corr.values, d.dec_error.values**2,
            d.dec_pmra_corr.values, d.dec_pmdec_corr.values],
        [d.ra_pmra_corr.values, d.dec_pmra_corr.values,
            d.pmra_error.values**2, d.pmra_pmdec_corr.values],
        [d.ra_pmdec_corr.values, d.dec_pmdec_corr.values,
            d.pmra_pmdec_corr.values, d.pmdec_error.values**2]
                    ])
    labels=["$\mathrm{RA~[degrees]}$", "$\mathrm{dec~[degrees]}$",
            "$\mu_{\mathrm{RA}}\mathrm{~[mas/year]}$",
            "$\mu_{\mathrm{dec}}\mathrm{~[mas/year]}$"]

    if parallax:
        mus = np.array([d.ra.values, d.dec.values, d.pmra.values,
                        d.pmdec.values, d.parallax.values])
        C = np.matrix([
            [d.ra_error.values**2, d.ra_dec_corr.values,
                d.ra_pmra_corr.values, d.ra_pmdec_corr.values,
                d.ra_parallax_corr.values],
            [d.ra_dec_corr.values, d.dec_error.values**2,
                d.dec_pmra_corr.values, d.dec_pmdec_corr.values,
                d.dec_parallax_corr.values],
            [d.ra_pmra_corr.values, d.dec_pmra_corr.values,
                d.pmra_error.values**2, d.pmra_pmdec_corr.values,
                d.parallax_pmra_corr.values],
            [d.ra_pmdec_corr.values, d.dec_pmdec_corr.values,
                d.pmra_pmdec_corr.values, d.pmdec_error.values**2,
                d.parallax_pmdec_corr.values],
            [d.ra_parallax_corr.values, d.dec_parallax_corr.values,
                d.parallax_pmra_corr.values,
                d.parallax_pmdec_corr.values,
                d.parallax_error.values**2]
                    ])
        labels=["$\mathrm{RA~[degrees]}$", "$\mathrm{dec~[degrees]}$",
                "$\mu_{\mathrm{RA}}\mathrm{~[mas/year]}$",
                "$\mu_{\mathrm{dec}}\mathrm{~[mas/year]}$",
                "$\pi\mathrm{~[mas]}$"]

    # Sample from the covariance matrix.
    corr_samps = np.random.multivariate_normal(mus, C, size=N).T

    if plot:
        figure = corner.corner(corr_samps.T, labels=labels)
        if parallax:
            plt.savefig("covariances_parallax_{}".format(d.kepid.values[i]))
        else:
            plt.savefig("covariances_{}".format(d.kepid.values[i]))

    return corr_samps

if __name__ == "__main__":

    # Load the csv of the targets, made with the action_time.ipynb notebook.
    mc = pd.read_csv("data/mcquillan_stars_with_vertical_action.csv")
    N = 10000

    # Generate samples with covariances for Gaia parameters for each star.
    nstars = np.shape(mc)[0]
    for i in range(nstars):
        samps = gen_sample_set(mc.iloc[i], N)
