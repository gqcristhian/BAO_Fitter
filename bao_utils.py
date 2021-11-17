import numpy as np
from numpy.linalg import inv
import sys

def set_bounds_and_masking(pk_data, cov, num_datasets, settings):
    settings["ind_keep"] = []
    settings["ind"] = []
    # set bounds for P(k)
    for i in range(0, num_datasets):
        # define full k-range and k-range for bao-analysis
        k = pk_data["{0}".format(i)]['k']
        pk_data["{0}".format(i)]['k'] = k[(k >= settings['kmin']) & (k <= settings['kmax'])]
        print("number of data points per multipole:", len(pk_data["{0}".format(i)]['k']), "for dataset", i)
        # get indices of elements not part of the analysis (k's to exclude)
        common_elements = np.in1d(k,pk_data["{0}".format(i)]['k'])
        common_elements = common_elements.tolist()
        indices_remove = [i for i, x in enumerate(common_elements) if x == False]
        if settings['bao_analysis_type'] == 'isotropic':
            # settings pk outside k-range to zero so they do not contribute
            for mult in settings['multipoles_indices']:
                pk = pk_data["{0}".format(i)] # pk dictionary for a given dataset
                pk['pk'] = pk['pk'][common_elements] # set pk-multipole elements outside k-range equal to zero
        if settings['bao_analysis_type'] == 'anisotropic':
            # settings pk multipoles elements to zero so they do not contribute
            for mult in settings['multipoles_indices']:
                pk = pk_data["{0}".format(i)] # pk dictionary for a given dataset
                pk["pk{0}".format(mult)] = pk["pk{0}".format(mult)][common_elements] # set pk-multipole elements outside k-range equal to zero
            for mult in [0,2,4]:
                if mult not in settings['multipoles_indices']:
                    pk["pk{0}".format(mult)] = []
        pk_data["{0}".format(i)] = pk # overwrite original pk_data dictionary for the i-th dataset
        print(pk_data["{0}".format(i)])
        # get indices to keep and total indices
        indices_keep = [i for i, x in enumerate(common_elements) if x == True]
        indices_elements = [i for i, x in enumerate(common_elements)]
        # storing indices into settings
        settings["ind_keep"].append(indices_keep)
        settings["ind"].append(indices_elements)
    # now continue with the covariance matrix
    inv_cov = {}
    for i in range(0, num_datasets): #creating cov[i] dictionaries for each dataset
        number = '%d' % i
        inv_cov[number] = {}
    for i in range(0, num_datasets):
        cov_matrix = cov["{0}".format(i)]
        # 1) here we will remove blocks of matrices if some multipoles are not included in the calculation
        ignore_multipoles = list(set([0,2,4]) - set(settings['multipoles_indices']))
        if len(ignore_multipoles) == 0: # all multipoles
            pass
        elif len(ignore_multipoles) == 1: # no hexadecapole term
            size = len(k)*2
            cov_matrix = cov_matrix[0:size,0:size]
        elif len(ignore_multipoles) == 2: # no hexadecapole and quadrupole terms
            size = len(k)
            cov_matrix = cov_matrix[0:size,0:size]
        else:
            print("Error: check set_bounds_and_masking() function (1)")
            sys.exit()
        # 2) now for we remove rows and columns corresponding to k elements that are outside the analysis k-range
        if len(ignore_multipoles) == 0: # all multipoles
            mask = np.concatenate((common_elements, common_elements, common_elements), axis=None)
        elif len(ignore_multipoles) == 1: # no hexadecapole term
            mask = np.concatenate((common_elements, common_elements), axis=None)
        elif len(ignore_multipoles) == 2: # no hexadecapole and quadrupole terms
            mask = common_elements
        else:
            print("Error: check set_bounds_and_masking() function (2)")
            sys.exit()
        cov_matrix = cov_matrix[mask,:][:,mask]
        inv_cov["{0}".format(i)] = inv(cov_matrix)

    return pk_data, inv_cov

def compare_theory_iso_data(data, index, pk, settings):
    diff = pk - data['pk']
    return diff

def compare_theory_ani_data(data, index, pk0, pk2, pk4, settings):
    # concatenating pk data
    if settings['multipoles_indices'] == [0]:
        data_pk = np.array(data['pk0'])
        model = np.array(pk0)
    if settings['multipoles_indices'] == [0, 2]:
        data_pk = np.concatenate((data['pk0'], data['pk2']), axis=None)
        model = np.concatenate((pk0, pk2), axis=None)
    if settings['multipoles_indices'] == [0, 2, 4]:
        data_pk = np.concatenate((data['pk0'], data['pk2'], data['pk4']), axis=None)
        model = np.concatenate((pk0, pk2, pk4), axis=None)
    diff = model - data_pk
    return diff

def delete_keys_from_dict(d, to_delete):
    if isinstance(to_delete, str):
        to_delete = [to_delete]
    if isinstance(d, dict):
        for single_to_delete in set(to_delete):
            if single_to_delete in d:
                del d[single_to_delete]
        for k, v in d.items():
            delete_keys_from_dict(v, to_delete)
    elif isinstance(d, list):
        for i in d:
            delete_keys_from_dict(i, to_delete)
    return d
