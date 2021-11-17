import numpy as np
import bao_utils

# ISOTROPIC BAO MODEL FROM LAUREN ANDERSON ET AL. (ARXIV 1312.4877)

def get_shifted_model(parameters, k, templates, settings):
    ''' Calculate a model including a shift given by alpha '''
    Pk = get_smooth_model(parameters, k, templates, settings) *\
     (1. + (templates['os_model'](k/parameters['alpha']['value']) - 1.)*Propagator(k, parameters))
    return Pk

def get_smooth_model(parameters, k, templates, settings):
    ''' Combine a noBAO model with polynomials and linear bias '''
    # calculating polynomial terms to fit broad-band
    polynomials = polynomial_terms(parameters, k, settings)
    # calculating Fingers of God term
    FoG = Fingers_of_God(k, parameters)
    FoG=1
    # calculating smoothed power spectrum
    return parameters['bias']['value'] ** 2 * templates['noBAO'](k) * FoG + polynomials

def polynomial_terms(parameters, x, settings):
    """
    Add polynomial terms to fit BAO broad-band.
    """
    # we add polynomial terms according to the number of polynomial terms set in inputs.py
    polynomials = 0
    for i in range(1,settings['poly_terms']+1):
        polynomials += parameters[f'a_0{i}']['value'] * x ** (settings['max_poly_power'] + 1 - i)

    return polynomials
def Fingers_of_God(k, parameters):
    """
    Calculates the Fingers of God term.
    """
    FoG = 1.0 / (1.0 + k ** 2 * parameters['Sigma_fog']['value'] ** 2 / 2.0 ) ** 2
    return FoG

def Propagator(k, parameters):
    """
    Calculates the propagator.
    """
    propagator = np.exp( -0.5 * k ** 2 * parameters['Sigma_NL']['value'] ** 2)
    return propagator

################################################################################

def lnprior(parameters, settings):
    lp = 0.
    # flat priors
    if not (parameters['bias']['value']>=parameters['bias']['low_lim'] and parameters['bias']['value']<=parameters['bias']['upp_lim']):
        return -np.inf, False
    if not (parameters['Sigma_fog']['value']>=parameters['Sigma_fog']['low_lim'] and parameters['Sigma_fog']['value']<=parameters['Sigma_fog']['upp_lim']):
        return -np.inf, False
    if not (parameters['Sigma_NL']['value']>=parameters['Sigma_NL']['low_lim'] and parameters['Sigma_NL']['value']<=parameters['Sigma_NL']['upp_lim']):
        return -np.inf, False
    if not (parameters['alpha']['value']>=parameters['alpha']['low_lim'] and parameters['alpha']['value']<=parameters['alpha']['upp_lim']):
        return -np.inf, False
    #gaussian prior on Sigma_fog
#    mu = 1.
#    sigma = 3.
#    lp += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(parameters['Sigma_fog']['value']-mu)**2/sigma**2
    return lp, True

def calc_chi2_pk(parameters, data, templates, func, settings):
    ''' Compares the model with the data '''
    chi2 = 0.
    logprior, within_priors = lnprior(parameters, settings)
    chi2 += -0.5*logprior
    if within_priors:
        # Loop over all datasets which are fit together
        for i in range(0,settings["num_datasets"]):
            pk = func(parameters, data[i]['k'], templates, settings)
            diff = bao_utils.compare_theory_iso_data(data[i], i, pk, settings)
            chi2 += np.dot(diff,np.dot(data[i]['cov_inv'],diff))
        return chi2
    else:
        return chi2
