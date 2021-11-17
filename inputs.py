import sys, os
import numpy as np
import itertools
import nbodykit
import pickle
import argparse
import configparser
import ast

################################################################################
######## FIRST PART: SET-UP PARAMETERS, SETTINGS (CHANGE IN inifiles/) #########
################################################################################
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = sys.argv[1]
inifile = dir_path + f'/inifiles/{filename}'
CONFIG = configparser.RawConfigParser()
CONFIG.read(inifile)

#base directories
chains_name = CONFIG.get('path', 'chains_name')
data_path = CONFIG.get('path', 'data_path')
output_path = CONFIG.get('path', 'output_path')

# DEFINE INPUT PARAMETERS
#template cosmology
Omega0_cdm = float(CONFIG.get('cosmology', 'Omega0_cdm'))
Omega0_b = float(CONFIG.get('cosmology', 'Omega0_b'))
h = float(CONFIG.get('cosmology', 'h'))

#fitting template options
bao_analysis_type = CONFIG.get('power_spectrum', 'bao_analysis_type')
load_pk = CONFIG.get('power_spectrum', 'load_pk')
path_pk = ast.literal_eval(CONFIG.get('power_spectrum', 'path_pk'))
smooth_method = CONFIG.get('power_spectrum', 'smooth_method')
reconstruction = CONFIG.get('power_spectrum', 'reconstruction')
num_poly_terms = int(CONFIG.get('power_spectrum', 'num_poly_terms_per_multipole'))
max_polynomial_power = int(CONFIG.get('power_spectrum', 'max_polynomial_power'))
multipoles = [int(item) for item in CONFIG.get('power_spectrum', 'multipoles').split()] # multipoles list to be fitted

# ISOTROPIC
if bao_analysis_type == 'isotropic':
    alpha = [float(item) for item in CONFIG.get('bao_parameters', 'alpha').split()]
    Sigma_NL = [float(item) for item in CONFIG.get('bao_parameters', 'Sigma_NL').split()]
elif bao_analysis_type == 'anisotropic':
# ANISOTROPIC
    beta = [float(item) for item in CONFIG.get('bao_parameters', 'beta').split()] # f/bias
    Sigma_NL_par = [float(item) for item in CONFIG.get('bao_parameters', 'Sigma_NL_par').split()]
    Sigma_NL_per = [float(item) for item in CONFIG.get('bao_parameters', 'Sigma_NL_per').split()]
    alpha_par = [float(item) for item in CONFIG.get('bao_parameters', 'alpha_par').split()]
    alpha_per = [float(item) for item in CONFIG.get('bao_parameters', 'alpha_per').split()]
# OTHER BAO PARAMETERS
bias = [float(item) for item in CONFIG.get('bao_parameters', 'bias').split()]
Sigma_fog = [float(item) for item in CONFIG.get('bao_parameters', 'Sigma_fog').split()]
A_ell = {}
for mult in multipoles:
    for poly in range(1,num_poly_terms+1):
        A_ell[f'a_{mult}{poly}'] = [float(item) for item in CONFIG.get('bao_parameters', f'a_{mult}{poly}').split()]
Sigma_smooth = float(CONFIG.get('bao_parameters', 'Sigma_smooth'))

# ranges for bao analysis
kmin = float(CONFIG.get('ranges', 'kmin'))
kmax = float(CONFIG.get('ranges', 'kmax'))
nmu = int(CONFIG.get('ranges', 'nmu'))
z_template = float(CONFIG.get('ranges', 'z_template'))

# mcmc settings
if CONFIG.get('mcmc', 'mpi') == 'True':
    mpi = True
elif CONFIG.get('mcmc', 'mpi') == 'False':
    mpi = False
Nchains = int(CONFIG.get('mcmc', 'Nchains'))
nwalkers = int(CONFIG.get('mcmc', 'nwalkers'))
ichaincheck = int(CONFIG.get('mcmc', 'ichaincheck'))
minlength = int(CONFIG.get('mcmc', 'minlength'))
epsilon = float(CONFIG.get('mcmc', 'epsilon'))

#datasets
pk_datafiles = ast.literal_eval(CONFIG.get('datasets', 'pk_datafiles'))
cov_datafiles = ast.literal_eval(CONFIG.get('datasets', 'cov_datafiles'))

################################################################################
########## SECOND PART: DEFINITIONS OF FUNCTIONS (NO NEED TO MODIFY) ###########
################################################################################

# GENERAL SETUP
def setup():

    #SETUP: TEMPLATE COSMOLOGY
    cosmo = nbodykit.lab.cosmology.Cosmology(Omega0_cdm=Omega0_cdm, Omega0_b=Omega0_b, h=h)

    # SETUP: POWER SPECTRUM
    num_datasets = len(pk_datafiles)
    pk_data = {}
    for i in range(0, num_datasets): #creating pk_data[i] dictionaries for each dataset
        number = '%d' % i
        pk_data[number] = {}
        pk_data[number]['k'] = []
        if bao_analysis_type == 'isotropic':
            pk_data[number]['pk'] = []
        if bao_analysis_type == 'anisotropic':
            pk_data[number]['pk0'] = []
            pk_data[number]['pk2'] = []
            pk_data[number]['pk4'] = []

    dataset_index = 0
    pk_files =  [data_path + pk for pk in pk_datafiles] # pk_files is a list of paths to each P(k)
    for pk_file in pk_files:
        mydata = open(pk_file, 'r')
        lines=mydata.readlines()[1:]
        for i in lines:
            pk_data["{0}".format(dataset_index)]['k'].append(i.split()[0])
        pk_data["{0}".format(dataset_index)]['k'] = np.array([float(item) for item in pk_data["{0}".format(dataset_index)]['k']])
        if bao_analysis_type == 'isotropic':
            for i in lines:
                pk_data["{0}".format(dataset_index)]['pk'].append(i.split()[1])
            pk_data["{0}".format(dataset_index)]['pk'] = np.array([float(item) for item in pk_data["{0}".format(dataset_index)]['pk']])
        if bao_analysis_type == 'anisotropic':
            column = 1
            for j in [0,2,4]:
                for i in lines:
                    pk_data["{0}".format(dataset_index)]["pk{0}".format(j)].append(i.split()[column])
                column += 1
                pk_data["{0}".format(dataset_index)]["pk{0}".format(j)] = np.array([float(item) for item in pk_data["{0}".format(dataset_index)]["pk{0}".format(j)]])
        dataset_index += 1

    # SETUP: COVARIANCE MATRIX
    cov_files =  [data_path + cov for cov in cov_datafiles] # pk_files is a list of paths to each P(k)
    cov = {}
    for i in range(0, num_datasets): #creating cov[i] dictionaries for each dataset
        number = '%d' % i
        cov[number] = {}
    dataset_index = 0
    for cov_file in cov_files:
        cov["{0}".format(dataset_index)] = np.loadtxt(cov_file)
        dataset_index += 1

    # SETUP: DEFINE DICTIONARY WITH INITIAL SETTINGS (to be extended...)
    settings = {
    "mpi": mpi,
    "chains_name": chains_name,
    "output_dir": output_path,
    "data_path": data_path,
    "num_datasets": num_datasets,
    "cosmo": cosmo,
    "kmin": kmin,
    "kmax": kmax,
    "z_template": z_template,
    "nmu" : nmu,
    "murange" : np.linspace(0., 1., nmu),
    "bao_analysis_type" : bao_analysis_type,
    "load_pk": load_pk,
    "path_pk": path_pk,
    "smooth_method": smooth_method,
    "reconstruction": reconstruction,
    "max_poly_power": max_polynomial_power,
    "poly_terms": num_poly_terms,
    "multipoles_indices": multipoles,
    "Nchains": Nchains,
    "nwalkers": nwalkers,
    "ichaincheck": ichaincheck,
    "minlength": minlength,
    "epsilon": epsilon
    }

    return settings, pk_data, cov

# DEFINE MODEL PARAMETERS BASED ON INPUT SETTINGS
def model_parameters(settings):
    # define dictionary
    params = {}
    # parameter to count number of model parameters
    num_params = 0

    # adding bias parameter
    params['bias'] = {'value': bias[0], 'low_lim': bias[1], 'upp_lim': bias[2], 'step_size': bias[3]}
    num_params += 1

    # adding Fingers of God parameters
    params['Sigma_fog'] = {'value': Sigma_fog[0], 'low_lim': Sigma_fog[1], 'upp_lim': Sigma_fog[2], 'step_size': Sigma_fog[3]}
    num_params += 1

    # adding bao model parameters
    if settings['bao_analysis_type'] == 'isotropic':
        # propagator parameters
        params['Sigma_NL'] = {'value': Sigma_NL[0], 'low_lim': Sigma_NL[1], 'upp_lim': Sigma_NL[2], 'step_size': Sigma_NL[3]}
        params['alpha'] = {'value': alpha[0], 'low_lim': alpha[1], 'upp_lim': alpha[2], 'step_size': alpha[3]}
        num_params += 2
    elif settings['bao_analysis_type'] == 'anisotropic':
        # kaiser term parameters
        params['beta'] = {'value': beta[0], 'low_lim': beta[1], 'upp_lim': beta[2], 'step_size': beta[3]}
        num_params += 1
        settings['Sigma_smooth'] = Sigma_smooth
        # propagator parameters
        params['Sigma_NL_par'] = {'value': Sigma_NL_par[0], 'low_lim': Sigma_NL_par[1], 'upp_lim': Sigma_NL_par[2], 'step_size': Sigma_NL_par[3]}
        params['Sigma_NL_per'] = {'value': Sigma_NL_per[0], 'low_lim': Sigma_NL_per[1], 'upp_lim': Sigma_NL_per[2], 'step_size': Sigma_NL_per[3]}
        params['alpha_par'] = {'value': alpha_par[0], 'low_lim': alpha_par[1], 'upp_lim': alpha_par[2], 'step_size': alpha_par[3]}
        params['alpha_per'] = {'value': alpha_per[0], 'low_lim': alpha_per[1], 'upp_lim': alpha_per[2], 'step_size': alpha_per[3]}
        num_params += 4

    # adding polynomial terms to parameters
    for mult in multipoles:
        for poly in range(1,num_poly_terms+1):
            params[f'a_{mult}{poly}'] = {'value': A_ell[f'a_{mult}{poly}'][0], 'low_lim': A_ell[f'a_{mult}{poly}'][1],\
                                         'upp_lim': A_ell[f'a_{mult}{poly}'][2], 'step_size': A_ell[f'a_{mult}{poly}'][3]}
            num_params += 1

    # adding num_params to settings
    settings["num_params"] = num_params

    return params
