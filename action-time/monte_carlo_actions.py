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

    # Sample from distance posterior.
    d_samps = np.mean([d.r_lo, d.r_hi])*np.random.randn(N) + d.r_est
    rv_samps = d.radial_velocity_error*np.random.randn(N) + d.radial_velocity

    # Make a corner plot and histograms of the samples.
    if plot:
        figure = corner.corner(corr_samps.T, labels=labels)
        if parallax:
            plt.savefig("covariances_parallax_{}".format(d.kepid))
            plt.close()
        else:
            plt.savefig("covariances_{}".format(d.kepid))
            plt.close()

        plt.hist(d_samps, 100)
        plt.savefig("d_samps_{}".format(d.kepid))
        plt.xlabel("Distance [kpc]")
        plt.close()

        plt.hist(rv_samps, 100)
        plt.savefig("rv_samps_{}".format(d.kepid))
        plt.xlabel("Radial velocity [km/s]")
        plt.close()

    return corr_samps, d_samps, rv_samps


def calc_actions_from_samples(Nsamps, samples, d_samps, rv_samps, plot=False):
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
    plot: (bool)
        If true a corner plot of the action samples will be created for a
        randomly chosen star.

    returns:
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
    """
    assert Nsamps <= len(d_samps), "Nsamps must not exceed the number of " \
        "available samples"

    ra_samps = samples[0, :]
    dec_samps = samples[1, :]
    pmra_samps = samples[2, :]
    pmdec_samps = samples[3, :]

    # R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, jR, lz, jz
    action_samps = np.zeros((Nsamps, 9))  # actions returns 9 values.

    for j in tqdm(range(Nsamps)):
        action_samps[j, :] = action(ra_samps[j], dec_samps[j], d_samps[j],
                                    pmra_samps[j],  pmdec_samps[j],
                                    rv_samps[j])

    if plot:

        labels = ["$R~[\mathrm{kpc}]$", "$\phi~[\mathrm{rad}]$",
                  "$Z~[\mathrm{kpc}]$", "$v_R~[\mathrm{kms}]$",
                  "$v_T~[\mathrm{kms}]$", "$v_z~[\mathrm{kms}]$", "$j_R$",
                  "$l_z$", "$j_z$"]
        figure = corner.corner(action_samps, labels=labels)
        plt.savefig("action_covariances_star_{}".format(i))
        plt.close()

    return action_samps


def stats_from_samps(samps):
    """ Calculate medians and uncertainties from a set of samples,
    samps: (array)
        1d array with samples.
    returns:
    =======
    samps: (1d array)
     Just the vertical action samples, again.
     (float)
     The median of the jz_samples.
    err: (float)
     The standard deviation of the jz_samples.
    errp: (float)
     The upper uncertainty on jz.
    errm: (float)
        The lower uncertainty on jz.
    """

    mu, std = np.median(samps), np.std(samps)
    upper = np.percentile(samps, 84)
    lower = np.percentile(samps, 16)
    errp, errm = upper - mu, mu - lower
    return mu, std, errp, errm


if __name__ == "__main__":

    # Load the csv of the targets, made with the action_time.ipynb notebook.
    mc = pd.read_csv("data/mcquillan_stars_with_vertical_action.csv")
    N = 100

    # Generate samples with covariances of Gaia parameters for each star.
    nstars = np.shape(mc)[0]

    R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, jR, lz, jz = \
        [np.zeros(nstars) for i in range(9)]
    R_err, phi_err, z_err, vR_err, vT_err, vz_err, jR_err, lz_err, jz_err = \
        [np.zeros(nstars) for i in range(9)]

    # Calculate actions for N samples
    for i in range(nstars):
        print("star", i, "of", nstars)
        samps, d_samps, rv_samps = gen_sample_set(mc.iloc[i], N)

        print("Calculating action samples for star", i)
        action_samps = calc_actions_from_samples(N, samps, d_samps, rv_samps)

        # Save samples in HDF5 file
        sample_df = pd.DataFrame(dict({"RA_deg": samps[0, :],
                                       "dec_deg": samps[1, :],
                                       "pmRA_masyr": samps[2, :],
                                       "pmdec_masyr": samps[3, :],
                                       "distance_kpc": d_samps,
                                       "RV_kms": rv_samps,
                                       "R_kpc": action_samps[:, 0],
                                       "phi_rad": action_samps[:, 1],
                                       "z_kpc": action_samps[:, 2],
                                       "vR_kms": action_samps[:, 3],
                                       "vT_kms": action_samps[:, 4],
                                       "vz_kms": action_samps[:, 5],
                                       "jR": action_samps[:, 6],
                                       "lz": action_samps[:, 7],
                                       "jz": action_samps[:, 8],
                                       }))

        store_export = pd.HDFStore("data/samples/{}_samples.h5"
                                   .format(mc.kepid.values[i]))
        store_export.append("samples", sample_df,
                            data_columns=sample_df.columns)
        store_export.close()

        # R_kpc[i], R_err[i], _, _ = stats_from_samps(action_samps[:, 0])
        # phi_rad[i], phi_err[i], _, _ = stats_from_samps(action_samps[:, 1])
        # z_kpc[i], z_err[i], _, _ = stats_from_samps(action_samps[:, 2])
        # vR_kms[i], vR_err[i], _, _ = stats_from_samps(action_samps[:, 3])
        # vT_kms[i], vT_err[i], _, _ = stats_from_samps(action_samps[:, 4])
        # vz_kms[i], vz_err[i], _, _ = stats_from_samps(action_samps[:, 5])
        # jR[i], jR_err[i], _, _ = stats_from_samps(action_samps[:, 6])
        # lz[i], lz_err[i], _, _ = stats_from_samps(action_samps[:, 7])
        # jz[i], jz_err[i], _, _ = stats_from_samps(action_samps[:, 8])

    # # Save results back into the dataframe
    # mc["R_kpc"], mc["R_err"] = R_kpc, R_err
    # mc["phi_rad"], mc["phi_err"] = phi_rad, phi_err
    # mc["z_kpc"], mc["z_err"] = z_kpc, z_err
    # mc["vR_kms"], mc["vR_err"] = vR_kms, vR_err
    # mc["vT_kms"], mc["vT_err"] = vT_kms, vT_err
    # mc["vz_kms"], mc["vz_err"] = vz_kms, vz_err
    # mc["jR"], mc["jR_err"] = jR, jR_err
    # mc["lz"], mc["lz_err"] = lz, lz_err
    # mc["jz"], mc["jz_err"] = jz, jz_err
    # mc.to_csv("data/mcquillan_stars_with_jz_uncerts.csv")
