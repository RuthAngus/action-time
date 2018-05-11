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


def construct_matrix(means, sigmas, covs):
    """
    Construct a covariance matrix.
    params:
    ======
    means: (array)
        An array containing the value means
    sigmas: (array)
        An array containing the variances (diagonal elements).
    covs: (array)
        An array containing the covariances between parameters from top left
        to bottom right.
    """
    # Make sure the arrays are all the correct shape.
    N = len(means)
    print(N, len(sigmas))
    print(np.shape(covs), (N-1, N))
    assert N == len(sigmas)
    assert np.shape(covs) == (N-1, N)

    C = np.matrix([)

    return cov


def gen_sample_set(d, N):
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
    print(d.keys())

    means = np.array([1, 2, 3])
    sigmas = np.array([.1, .2, .3])
    aa, bb, cc = .1, .2, .3
    ab, ac, bc = .4, .5, .6
    covs_a = np.array([aa, ab, ac])
    covs_b = np.array([ab, bb, bc])
    covs_c = np.array([ac, bc, cc])
    cov = construct_matrix(means, sigmas, covs)
    assert 0

    # assign covariance variables
    ra_dec = d.tgas_ra_dec_corr.values
    ra_pmra = d.tgas_ra_pmra_corr.values
    ra_pmdec = d.tgas_ra_pmdec_corr.values
    dec_pmra = d.tgas_dec_pmra_corr.values
    dec_pmdec = d.tgas_dec_pmdec_corr.values
    pmra_pmdec = d.tgas_pmra_pmdec_corr.values

    mus = np.array([d.ra.values, d.dec.values, d.pmra.values, d.pmdec.values])
    C = np.matrix([[ra_err**2, ra_dec, ra_pmra, ra_pmdec],
                   [ra_dec, dec_err**2, dec_pmra, dec_pmdec],
                   [ra_pmra, dec_pmra, pmra_err**2, pmra_pmdec],
                   [ra_pmdec, dec_pmdec, pmra_pmdec, pmdec_err**2]
                   ])

    # Sample from the covariance matrix.
    corr_samps = np.random.multivariate_normal(mus, C, size=N).T
    ra_samps, dec_samps, pmra_samps, pmdec_samps = corr_samps

    # distance_samps
    # rv_samps

    return teff_samps, feh_samps, logg_samps, ra_samps, dec_samps, d_samps, \
        pmra_samps, pmdec_samps, rv_samps


# def sample_from_distance_posterior():

if __name__ == "__main__":
    dat = Table.read('data/kepler_dr2_4arcsec_updated.fits', format='fits')
    d = dat.to_pandas()
    d["relative_parallax_err"] = d.parallax_error.values / d.parallax.values
    gen_sample_set(d, 10000)
