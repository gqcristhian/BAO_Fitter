import sys
#importing mcmc modules
import emcee
from schwimmbad import MPIPool
#importing local python modules
import inputs
import bao_utils
import bao_template
import bao_model_iso
import bao_model_ani
import mcmc
#importing extra python modules
import numpy as np
import copy
import scipy.optimize as op

################################################################################
######### FIRST PART: DEFINING PARAMETERS, DATA AND ANALYSIS SETTINGS ##########
################################################################################

settings, pk_data, cov = inputs.setup() # saving settings into a dictionary (see inputs module)
params = inputs.model_parameters(settings) # defining dictionary for bao model parameters
# printing some initial settings on the screen
print("\n##### FIRST MODULE RUN SUCCESSFULLY #####")
print("initial settings...")
print("chains name:", settings["chains_name"])
print("data directory:", settings["data_path"])
print("output directory:", settings["output_dir"])
print("kmin:", settings["kmin"])
print("kmax:", settings["kmax"])
print("bao analysis:", settings["bao_analysis_type"])
print("reconstruction:", settings["reconstruction"])
print("number of polynomial terms:", settings["poly_terms"])
print("maximum power of polynomial terms:", settings["max_poly_power"])
if settings['bao_analysis_type'] == 'anisotropic':
    print("Sigma smoothing:", settings["Sigma_smooth"])
    print("multipoles to calculate:", settings["multipoles_indices"])
print("initial parameters: ", params)
print("...\n")

################################################################################
######### SECOND PART:  SETTING K-RANGE FOR ANALYSIS AND SETTING DATA ##########
################################################################################

# set range of k for bao-analysis (using kmax and kmin)
pk_data, cov_inv = bao_utils.set_bounds_and_masking(pk_data, cov, settings["num_datasets"], settings)
# append datasets information into a list of dictionaries called data
data = []
for i in range(0,settings["num_datasets"]):
    tag = str(i)
    if settings['bao_analysis_type'] == 'isotropic':
        data.append( { 'k': pk_data[tag]['k'], 'pk': pk_data[tag]['pk'],\
         'cov_inv': cov_inv[tag]}  )
    if settings['bao_analysis_type'] == 'anisotropic':
        data.append( { 'k': pk_data[tag]['k'],\
         'pk0': pk_data[tag]['pk0'], 'pk2': pk_data[tag]['pk2'], 'pk4': pk_data[tag]['pk4'],\
          'cov_inv': cov_inv[tag]}  )
print("##### SECOND MODULE RUN SUCCESSFULLY #####")
print("data restricted to kmin =", settings['kmin']," and kmax =", settings['kmax'])
print("...\n")

################################################################################
############### THIRD PART:  CALCULATING OSCILLATING BAO SIGNAL ################
################################################################################
##### ISOLATE THE BAO SIGNAL
# get linear power spectrum
k_template_range, Pk_lin = bao_template.Pk_template(settings)
# get smoothed linear P(k)
Pk_sm = bao_template.get_lin_smooth_model(k_template_range, Pk_lin, settings)
# calculate O(k)
os_model = bao_template.get_oscillation(k_template_range, Pk_lin, Pk_sm, settings)
print("##### THIRD MODULE RUN SUCCESSFULLY #####")
print("O(k) calculated!, using ", settings['smooth_method'], "method...")
print("...\n")

################################################################################
########### FOURTH PART: PERFORMING MCMC ANALYSIS FOR OUR BAO MODEL ############
################################################################################

# running mcmc
dim = len(params)
Nchains = settings["Nchains"]
nwalkers = settings["nwalkers"]
ichaincheck = settings["ichaincheck"]
minlength = settings["minlength"]
epsilon = settings["epsilon"]

labels = list(params.keys())
limits = copy.deepcopy(params)
limits = bao_utils.delete_keys_from_dict(limits,['value','step_size'])
initial_values = [val.get('value') for val in params.values()]

# get expected error from step_size key in dictionary
expected_error = []
for i in range(0,dim):
    expected_error.append(list(params.values())[i]['step_size'])

# Defining likelihood
def get_loglike(theta):
    value = [{'value': i} for i in theta]
    parameters = dict(zip(labels, value))
    for param in labels:
        parameters[param].update(limits[param])
    if settings['bao_analysis_type'] == 'isotropic':
        return -0.5*bao_model_iso.calc_chi2_pk(parameters, data, { 'noBAO': Pk_sm, 'os_model': os_model }, bao_model_iso.get_shifted_model, settings)
    if settings['bao_analysis_type'] == 'anisotropic':
        return -0.5*bao_model_ani.calc_chi2_mult(parameters, data, {'k_template_range': k_template_range, 'noBAO': Pk_sm(k_template_range), 'os_model': os_model(k_template_range) }, bao_model_ani.get_multipoles, settings)

# Set up the sampler.
pos=[]
list_of_samplers=[]

if settings["mpi"]==True:
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        for jj in range(0, Nchains):
            pos.append([initial_values + (2.*np.random.random_sample((dim,)) - 1.)*expected_error for i in range(nwalkers)])
            list_of_samplers.append(emcee.EnsembleSampler(nwalkers=nwalkers, ndim=dim, log_prob_fn=get_loglike, pool=pool))

        # Start MCMC
        print("Running MCMC... ")
        within_chain_var = np.zeros((Nchains, dim))
        mean_chain = np.zeros((Nchains, dim))
        scalereduction = np.arange(dim, dtype=np.float)
        scalereduction.fill(2.)

        itercounter = 0
        chainstep = minlength
        while any(abs(1. - scalereduction) > epsilon):
            itercounter += chainstep
            for jj in range(0, Nchains):
                for result in list_of_samplers[jj].sample(pos[jj], iterations=chainstep, store=True):
                    pos[jj] = result.coords # for emcee 2.2.1 we used result[0]
                # we do the convergence test on the second half of the current chain (itercounter/2)
                within_chain_var[jj], mean_chain[jj] = mcmc.prep_gelman_rubin(list_of_samplers[jj])
            scalereduction = mcmc.gelman_rubin_convergence(within_chain_var, mean_chain, int(itercounter/2))
            print("scalereduction = ", scalereduction)
            chainstep = ichaincheck
        # Investigate the chain and get mean values
        averages = mcmc.inspect_chain(list_of_samplers, settings, labels)
        print("##### FOURTH MODULE RUN SUCCESSFULLY #####")
        print("please check output at", settings["output_dir"])
        # Get best-fit model values by using the mean value of the sample distributions as starting point for the scipy optimizer
        nll = lambda *args: -get_loglike(*args)
        result = op.minimize(nll, list(averages))
        best_fit_parameters = result.x.tolist()
        print("Maximum likelihood estimates:")
        print("bf_chi2 = ", 2*nll(best_fit_parameters))
elif settings["mpi"]==False:
    for jj in range(0, Nchains):
        pos.append([initial_values + (2.*np.random.random_sample((dim,)) - 1.)*expected_error for i in range(nwalkers)])
        list_of_samplers.append(emcee.EnsembleSampler(nwalkers=nwalkers, ndim=dim, log_prob_fn=get_loglike))

    # Start MCMC
    print("Running MCMC... ")
    within_chain_var = np.zeros((Nchains, dim))
    mean_chain = np.zeros((Nchains, dim))
    scalereduction = np.arange(dim, dtype=np.float)
    scalereduction.fill(2.)

    itercounter = 0
    chainstep = minlength
    while any(abs(1. - scalereduction) > epsilon):
        itercounter += chainstep
        for jj in range(0, Nchains):
            for result in list_of_samplers[jj].sample(pos[jj], iterations=chainstep, store=True):
                pos[jj] = result.coords # for emcee 2.2.1 we used result[0]
            # we do the convergence test on the second half of the current chain (itercounter/2)
            within_chain_var[jj], mean_chain[jj] = mcmc.prep_gelman_rubin(list_of_samplers[jj])
        scalereduction = mcmc.gelman_rubin_convergence(within_chain_var, mean_chain, int(itercounter/2))
        print("scalereduction = ", scalereduction)
        chainstep = ichaincheck
    # Investigate the chain and get mean values
    averages = mcmc.inspect_chain(list_of_samplers, settings, labels)
    print("##### FOURTH MODULE RUN SUCCESSFULLY #####")
    print("please check output at", settings["output_dir"])
    # Get best-fit model values by using the mean value of the sample distributions as starting point for the scipy optimizer
    nll = lambda *args: -get_loglike(*args)
    result = op.minimize(nll, list(averages))
    best_fit_parameters = result.x.tolist()
    print("Maximum likelihood estimates:")
    print("bf_chi2 = ", 2*nll(best_fit_parameters))

################################################################################
################ FIFTH PART: SAVING SOME RESULTS AND SETTINGS  #################
################################################################################

# save best fit model
res = open(settings["output_dir"] + 'info/' + "BestFitModel_" + settings['chains_name'] + '.minimum', "w")
res.write(" -log(Like) =   {} \n".format(nll(best_fit_parameters)))
res.write("  chi-sq    =   {} \n\n".format(2*nll(best_fit_parameters)))
for i in range(0,len(best_fit_parameters)):
    res.write("{0}  {1}  {2} \n".format(int(i+1), float(best_fit_parameters[i]), str(labels[i]).ljust(20)))
res.close()
# save parameters ranges
res = open(settings["output_dir"] + 'info/' + "samples_" + settings['chains_name'] + '.ranges', "w")
for i in range(0,len(best_fit_parameters)):
    res.write("{0}  {1}  {2} \n".format(str(labels[i]).ljust(20), params[labels[i]]["low_lim"], params[labels[i]]["upp_lim"]))
res.close()
# save initial inputs file
res = open(settings["output_dir"]+'info/' + settings['chains_name'] + '.inputs', "w")
res.write("# initial settings" + "\n")
res.write("settings: " + str(settings))
res.write("parameters: " + str(params))
res.close()
# final message
print("code finished! take a look at the output folder.")
