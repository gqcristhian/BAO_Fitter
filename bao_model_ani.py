from numba import jit
import numpy as np
import bao_utils
from scipy.integrate import simps
import time
import sys

# ANISOTROPIC BAO MODEL FROM FLORIAN BEUTLER ET AL. (ARXIV 1607.03149)

def get_params_values(params, settings):
    bias = params['bias']['value']
    Sigma_fog = params['Sigma_fog']['value']
    beta = params['beta']['value']
    Sigma_NL_par = params['Sigma_NL_par']['value']
    Sigma_NL_per = params['Sigma_NL_per']['value']
    alpha_par = params['alpha_par']['value']
    alpha_per = params['alpha_per']['value']
    return bias, Sigma_fog, beta, Sigma_NL_par, Sigma_NL_per, alpha_par, alpha_per

def get_settings(settings):
    reconstruction = settings['reconstruction']
    Sigma_smooth = settings['Sigma_smooth']
    return reconstruction, Sigma_smooth

@jit(nopython=True, parallel = True) # Set "nopython" mode for best performance, equivalent to @njit
def get_shifted_model(pars, sets, x, y, k_template_range, Pk_sm_template, os_model):
    ''' Calculate a model including a shift given by alpha '''
    bias, Sigma_fog, beta, Sigma_NL_par, Sigma_NL_per, alpha_par, alpha_per = pars
    reconstruction, Sigma_smooth = sets
    Pkmu = np.zeros((len(x), len(y)))
    for k in range(0,len(x)):
        for mu in range(0,len(y)):
            # get dilated coordinates
            kp, mup = AP_scaling(x[k], y[mu], alpha_par, alpha_per)
            # get FoG term (sensitive to small scales)
            FoG = Fingers_of_God(kp, mup, Sigma_fog)
            # get Kaiser Term (sensitive to large scales)
            KT = Kaiser_term(kp, mup, beta, Sigma_smooth, reconstruction)
            # calculate Pk_nw by interpolation
            Pk_sm = bias ** 2 * KT * interp_nb(kp, k_template_range, Pk_sm_template) * FoG
            Ok = interp_nb(kp, k_template_range, os_model)
            Pkmu[k][mu] = Pk_sm * (1. + (Ok - 1.) * Propagator(kp, mup, Sigma_NL_par, Sigma_NL_per))
    return Pkmu

def get_multipoles(parameters, k, mu, templates, settings):
    # calculate P(k,mu) in dilated coordinates (AP rescaling is taken inside)
    pk2d = get_shifted_model(get_params_values(parameters, settings), get_settings(settings), k, mu, templates['k_template_range'], templates['noBAO'], templates['os_model'])
    # performing integration over mu to get multipoles and adding BB polynomials
    pk0 = simps(pk2d, mu, axis=1)
    if settings['multipoles_indices'] == [0]:
        # adding polynomial terms to fit broad-band
        pk0 += polynomial_terms(parameters, k, 0, settings)
        return pk0, np.zeros(len(pk0)), np.zeros(len(pk0))
    if settings['multipoles_indices'] == [0, 2]:
        pk2 = 3.0 * simps(pk2d * mu ** 2, mu, axis=1)
        pk2 = 2.5 * (pk2 - pk0)
        pk0 += polynomial_terms(parameters, k, 0, settings)
        pk2 += polynomial_terms(parameters, k, 2, settings)
        return pk0, pk2, np.zeros(len(pk0))
    if settings['multipoles_indices'] == [0, 2, 4]:
        pk2 = 3.0 * simps(pk2d * mu ** 2, mu, axis=1)
        pk4 = 1.125 * (35.0 * simps(pk2d * mu ** 4, mu, axis=1) - 10.0 * pk2 + 3.0 * pk0)
        pk2 = 2.5 * (pk2 - pk0)
        pk0 += polynomial_terms(parameters, k, 0, settings)
        pk2 += polynomial_terms(parameters, k, 2, settings)
        pk4 += polynomial_terms(parameters, k, 4, settings)
        return pk0, pk2, pk4

################################################################################

def polynomial_terms(parameters, x, multipole, settings):
    """
    Add polynomial terms to fit BAO broad-band.
    """
    # we add polynomial terms according to the number of polynomial terms set in inputs.py
    polynomials = 0
    for i in range(1,settings['poly_terms']+1):
        polynomials += parameters[f'a_{multipole}{i}']['value'] * x ** (settings['max_poly_power'] + 1 - i)

    return polynomials

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def Fingers_of_God(k, mu, Sigma_fog):
    """
    Calculates the Fingers of God term.
    """
    FoG = 1.0 / (1.0 + k ** 2 * mu ** 2 * Sigma_fog ** 2 / 2.0 ) ** 2
    return FoG

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def Kaiser_term(k, mu, beta, Sigma_smooth, reconstruction):
    """
    Calculates Kaiser term.
    """
    if reconstruction == 'ani':
        R = 1.0
    elif reconstruction == 'iso':
        R = 1.0 - np.exp( -(k * Sigma_smooth) ** 2 / 2.0 )
    KT = (1.0 + beta * mu ** 2 * R)**2
    return KT

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def Propagator(k, mu, Sigma_NL_par, Sigma_NL_per):
    """
    Calculates the propagator.
    """
    propagator = np.exp( -0.5 * k ** 2 * (mu ** 2 * Sigma_NL_par ** 2 + (1.0 - mu ** 2) * Sigma_NL_per ** 2) )
    return propagator

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def AP_scaling(k, mu, alpha_par, alpha_per):
    """
    Calculates Alcock-Paczynski scaling for k and mu.
    """
    F = alpha_par / alpha_per
    k_prime = k / alpha_per * (1.0 + mu ** 2 * (1.0 / F ** 2 - 1.0)) ** 0.5
    mu_prime = mu / F / (1.0 + mu ** 2 * (1.0 / F ** 2 - 1.0)) ** 0.5
    return k_prime, mu_prime

################################################################################

def lnprior(parameters, settings):
    lp = 0.
    # flat priors
    if not (parameters['bias']['value']>=parameters['bias']['low_lim'] and parameters['bias']['value']<=parameters['bias']['upp_lim']):
        return -np.inf, False
    if not (parameters['Sigma_fog']['value']>=parameters['Sigma_fog']['low_lim'] and parameters['Sigma_fog']['value']<=parameters['Sigma_fog']['upp_lim']):
        return -np.inf, False
    if not (parameters['beta']['value']>=parameters['beta']['low_lim'] and parameters['beta']['value']<=parameters['beta']['upp_lim']):
        return -np.inf, False
    if not (parameters['Sigma_NL_par']['value']>=parameters['Sigma_NL_par']['low_lim'] and parameters['Sigma_NL_par']['value']<=parameters['Sigma_NL_par']['upp_lim']):
        return -np.inf, False
    if not (parameters['Sigma_NL_per']['value']>=parameters['Sigma_NL_per']['low_lim'] and parameters['Sigma_NL_per']['value']<=parameters['Sigma_NL_per']['upp_lim']):
        return -np.inf, False
    if not (parameters['alpha_par']['value']>=parameters['alpha_par']['low_lim'] and parameters['alpha_par']['value']<=parameters['alpha_par']['upp_lim']):
        return -np.inf, False
    if not (parameters['alpha_per']['value']>=parameters['alpha_per']['low_lim'] and parameters['alpha_per']['value']<=parameters['alpha_per']['upp_lim']):
        return -np.inf, False
    #gaussian prior on Sigma_fog
    #mu = 1.
    #sigma = 3.
    #lp += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(parameters['Sigma_fog']['value']-mu)**2/sigma**2
    #gaussian prior on Sigma_par
    #mu = 8.
    #sigma = 4.
    #lp += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(parameters['Sigma_NL_par']['value']-mu)**2/sigma**2
    #gaussian prior on Sigma_per
    #mu = 4.
    #sigma = 8.
    #lp += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(parameters['Sigma_NL_per']['value']-mu)**2/sigma**2
    return lp, True

def calc_chi2_mult(parameters, data, templates, func, settings):
    ''' Compares the model with the data '''
    chi2 = 0.
    logprior, within_priors = lnprior(parameters, settings)
    chi2 += -0.5*logprior
    if within_priors:
        # Loop over all datasets which are fit together
        for i in range(0,settings["num_datasets"]):
            pk0, pk2, pk4 = func(parameters, data[i]['k'], settings['murange'], templates, settings)
            diff = bao_utils.compare_theory_ani_data(data[i], i, pk0, pk2, pk4, settings)
            chi2 += np.dot(diff,np.dot(data[i]['cov_inv'],diff))
        return chi2
    else:
        return chi2

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def interp_nb(x_vals, x, y):
    return np.interp(x_vals, x, y)
