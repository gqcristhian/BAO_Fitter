import numpy as np
import scipy.optimize as op
import scipy.interpolate as interp
#importing nbodykit for template calculation
import nbodykit.lab
import nbodykit

# Defining template
def Pk_template(settings):
    # array of k where we are going to define the linear power spectrum
    kmin=0.005
    kmax=0.5
    k_template_range = np.arange(kmin, kmax, 0.001)

    #load P(k) (otherwise calculated object with nbodykit)
    if settings['load_pk'] == True:
        print("Pk_template: loading Pk_lin from file...")
        #read linear P(k) from file and append into lists
        mydata = open(settings['path_pk'], 'r')
        lines=mydata.readlines()[0:]
        k = []
        Pk_lin = []
        for i in lines:
            k.append(float(i.split()[0]))
            Pk_lin.append(float(i.split()[1]))
        mydata.close()
        #convert lists to arrays for k and P(k)
        k = np.array(k)
        Pk_lin = np.array(Pk_lin)
        #make sure we only use k in the range (0.001,0.5) and not beyond to create template for O(k)
        k_template_range = k[(k>=kmin) & (k<=kmax)]
        mask = np.in1d(k, k_template_range)
        Pk_lin = Pk_lin[mask]
        #create an interpolator for P(k)
        Pk_lin = interp.interp1d(k_template_range, Pk_lin)
    else:
        print("Pk_template: no Pk_lin given, calculating it using nbodykit...")
        Pk_lin = nbodykit.lab.cosmology.power.linear.LinearPower(settings['cosmo'], redshift=settings["z_template"], transfer='CLASS')
    #return interpolator objetcs
    return k_template_range, Pk_lin

def get_lin_smooth_model(k_template_range, Pk_lin, settings):
    ''' Get smoothed linear power spectrum '''
    # If using EH98 method, calculate non-wiggle power spectrum, otherwise set to None.
    if settings['smooth_method'] == 'EH98':
        print("get_lin_smooth_model: computing non-wiggle Eisenstein-Hu-1998 P(k)")
        Pk_nw = nbodykit.lab.cosmology.power.linear.LinearPower(settings['cosmo'], redshift=settings["z_template"], transfer='NoWiggleEisensteinHu')
        # fitting smoothed P(k)
        start = [1., 0., 0., 0., 0., 0.]
        result = op.minimize(calc_chi2_os, start, args=( { 'k': k_template_range, 'pk': Pk_lin(k_template_range) }, Pk_nw, smooth_EH98),
        method="Nelder-Mead", tol=1.0e-6, options={"maxiter": 1000000})
        Pk_sm = smooth_EH98(result["x"], k_template_range, Pk_nw)
        return interp.interp1d(k_template_range, Pk_sm)
    else:
        print("get_lin_smooth_model: Using smoothed P(k) from Hinton-2017")
        Pk_sm = smooth_hinton2017(k_template_range, Pk_lin(k_template_range))
        return interp.interp1d(k_template_range, Pk_sm)

# Oscillation feature
def get_oscillation(k_template_range, Pk_lin, Pk_sm, settings):
    ''' Get an oscillation only power spectrum '''
    oscillation_factor = Pk_lin(k_template_range)/Pk_sm(k_template_range)
    return interp.interp1d(k_template_range, oscillation_factor)

# Smooth P(k) approaches (Hinton2017 or EisensteinHu1998)
def calc_chi2_os(parameters, data, templates, func):
    ''' Compares the model with the data '''
    chi2 = 0.
    model = func(parameters, data['k'], templates)
    diff = (model - data['pk'])
    chi2 = np.sum( (diff / data['pk'])** 2 )
    return chi2

def smooth_EH98(parameters, k, Pk_nw):
    ''' Combine a noBAO model with polynomials and linear bias '''
    # calculating polynomial terms to fit broad-band
    polynomials = parameters[1]/k**3 + parameters[2]/k**2 + parameters[3]/k + parameters[4] + parameters[5]*k
    # calculating smoothed power spectrum
    return parameters[0] * parameters[0] * Pk_nw(k) + polynomials

# function code for smooth_hinton2017 taken from Barry
def smooth_hinton2017(ks, pk, degree=13, sigma=1, weight=0.5):
    """ Smooth power spectrum based on Hinton 2017 polynomial method """
    # logging.debug("Smoothing spectrum using Hinton 2017 method")
    log_ks = np.log(ks)
    log_pk = np.log(pk)
    index = np.argmax(pk)
    maxk2 = log_ks[index]
    gauss = np.exp(-0.5 * np.power(((log_ks - maxk2) / sigma), 2))
    w = np.ones(pk.size) - weight * gauss
    z = np.polyfit(log_ks, log_pk, degree, w=w)
    p = np.poly1d(z)
    polyval = p(log_ks)
    pk_smoothed = np.exp(polyval)
    return pk_smoothed
