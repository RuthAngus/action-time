"""
MONTE_CARLO_ACTIONS
===================
This code uses the covariances on RA, dec, pmra and pmdec to calculate actions
with uncertainties.
It also uses the Bailer-Jones posterior for distances.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import corner
from actions import action
from tqdm import tqdm

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

    mus = np.array([d.ra, d.dec, d.pmra, d.pmdec])
    C = np.matrix([
        [d.ra_error**2, d.ra_dec_corr, d.ra_pmra_corr, d.ra_pmdec_corr],
        [d.ra_dec_corr, d.dec_error**2, d.dec_pmra_corr, d.dec_pmdec_corr],
        [d.ra_pmra_corr, d.dec_pmra_corr, d.pmra_error**2, d.pmra_pmdec_corr],
        [d.ra_pmdec_corr, d.dec_pmdec_corr, d.pmra_pmdec_corr,
         d.pmdec_error**2]
                    ])
    labels=["$\mathrm{RA~[degrees]}$", "$\mathrm{dec~[degrees]}$",
            "$\mu_{\mathrm{RA}}\mathrm{~[mas/year]}$",
            "$\mu_{\mathrm{dec}}\mathrm{~[mas/year]}$"]

    if parallax:
        mus = np.array([d.ra, d.dec, d.pmra, d.pmdec, d.parallax])
        C = np.matrix([
            [d.ra_error**2, d.ra_dec_corr, d.ra_pmra_corr, d.ra_pmdec_corr,
             d.ra_parallax_corr],
            [d.ra_dec_corr, d.dec_error**2, d.dec_pmra_corr, d.dec_pmdec_corr,
             d.dec_parallax_corr],
            [d.ra_pmra_corr, d.dec_pmra_corr, d.pmra_error**2,
             d.pmra_pmdec_corr, d.parallax_pmra_corr],
            [d.ra_pmdec_corr, d.dec_pmdec_corr, d.pmra_pmdec_corr,
             d.pmdec_error**2, d.parallax_pmdec_corr],
            [d.ra_parallax_corr, d.dec_parallax_corr, d.parallax_pmra_corr,
             d.parallax_pmdec_corr,
             d.parallax_error**2]
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
            plt.savefig("covariances_parallax_{}".format(d.kepid))
        else:
            plt.savefig("covariances_{}".format(d.kepid))

    # Sample from distance posterior.
    d_samps = np.mean([d.r_lo, d.r_hi])*np.random.randn(N) + d.r_est
    rv_samps = d.radial_velocity_error*np.random.randn(N) + d.radial_velocity

    return corr_samps, d_samps, rv_samps


def calc_actions_from_samples(Nsamps, samples, d_samps, rv_samps):
    """
    Call Wilma's action function and calculate actions for each sample.
    params:
    ======
    Nsamps: (int)
        The number of action samples you want. Could be the same as the number
        of RA, dec, etc samples you generated or could be fewer.
    samples: (array)
        2d array of samples for RA, dec, pmra and pmdec. size = (4, Nsamps)
    d_samps: (array)
        1d array of distance samples. len = Nsamps.
    rv_samps: (array)
        1d array of radial velocity samples. len = Nsamps.
    return
    ======
    action_samps: (2d array)
        Array containing samples of each of the values calculated by
        actions.py. These are:
        R_kpc: Samples of the galactic radius in kpc.
        phi_rad: Samples of phi_rad FIXME: look up what this is!
        z_kpc: Samples of the height above the galactic plane.
        vR_kms: Samples of radial velocity -- check this.
        vT_kms: Samples of tangential velocity -- check this.
        vz_kms: Samples of vertical velocity.
        jR: Samples of Radial action.
        lz: Samples of ...
        jz: Samples of vertical action.
    jz_samps: (1d array)
        Just the vertical action samples, again.
    """
    assert Nsamps <= len(d_samps), "Nsamps must not exceed the number of " \
        "available samples"

    ra_samps = samples[0, :]
    dec_samps = samples[1, :]
    pmra_samps = samples[2, :]
    pmdec_samps = samples[3, :]

    # R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, jR, lz, jz
    action_samps = np.zeros((Nsamps, 9))  # actions returns 9 values.

    for i in range(Nsamps):
        action_samps[i, :] = action(ra_samps[i], dec_samps[i], d_samps[i],
                                    pmra_samps[i],  pmdec_samps[i],
                                    rv_samps[i])
    return action_samps, action_samps[:, -1]


if __name__ == "__main__":

    # Load the csv of the targets, made with the action_time.ipynb notebook.
    mc = pd.read_csv("data/mcquillan_stars_with_vertical_action.csv")
    N = 10000

    # Generate samples with covariances for Gaia parameters for each star.
    nstars = np.shape(mc)[0]
    for i in range(nstars):
        samps, d_samps, rv_samps = gen_sample_set(mc.iloc[i], N)
        action_samps, jz_samps = calc_actions_from_samples(10, samps, d_samps,
                                                           rv_samps)
        assert 0
