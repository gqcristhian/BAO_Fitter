import numpy as np
import matplotlib.pyplot as plt
import corner

def get_percentiles(chain, labels):
    ''' Calculate constraints and uncertainties from MCMC chain '''
    per = np.percentile(chain, [50., 15.86555, 84.13445, 2.2775, 97.7225], axis=0)
    per = np.array([per[0], per[0]-per[1], per[2]-per[0], per[0]-per[3], per[4]-per[0]])
    for i in range(0, len(per[0])):
        print("%s = %f +%f -%f +%f -%f" % (labels[i], per[0][i], per[1][i], per[2][i], per[3][i], per[4][i]))
    percentiles = dict(zip(labels, [ [x[i] for x in per] for i in range(0,len(per[0])) ] ))
    return percentiles

def get_averages(chain):
    ''' Calculate constraints and uncertainties from MCMC chain '''
    average = np.average(chain, axis=0)
    return average

def inspect_chain(list_of_samplers, settings, labels=[]):
    ''' Print chain properties '''
    output_path_mcmc = settings['output_dir'] + 'mcmc/'
    output_path_plots = settings['output_dir'] + 'plots/'

    Nchains = len(list_of_samplers)
    dim = list_of_samplers[0].chain.shape[2]
    if not labels:
        # set default labels
        labels = [('para_%i' % i) for i in range(0,dim)]

    mergedsamples = []
    for jj in range(0, Nchains):
        chain_length = list_of_samplers[jj].chain.shape[1]
        mergedsamples.extend(list_of_samplers[jj].chain[:, int(chain_length/2):, :].reshape((-1, dim)))

    # write out chain
    res = open(output_path_mcmc + "samples_" + settings["chains_name"] + ".dat", "w")
    res.write("#" + ", ".join(labels) + "\n")

    for row in mergedsamples:
        for el in row:
            res.write("%f " % el)
        res.write("\n")
    res.close()

    print("length of merged chain = ", len(mergedsamples))
    try:
        for jj in range(0, Nchains):
            print("Mean acceptance fraction for chain ", jj,": ", np.mean(list_of_samplers[jj].acceptance_fraction))
    except Exception as e:
        print("WARNING: %s" % str(e))
    try:
        for jj in range(0, Nchains):
            print("Autocorrelation time for chain ", jj,": ", list_of_samplers[jj].get_autocorr_time())
    except Exception as e:
        print("WARNING: %s" % str(e))

    fig, axes = plt.subplots(dim, 1, sharex=True, figsize=(8, 9))
    for i in range(0, dim):
        for jj in range(0, Nchains):
            axes[i].plot(list_of_samplers[jj].chain[:, :, i].T, alpha=0.4)
        #axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(labels[i])
    fig.tight_layout(h_pad=0.0)
    fig.savefig(output_path_plots + settings["chains_name"] + "time_series.png")

    try:
        return get_averages(mergedsamples)
    except Exception as e:
        print("WARNING: %s" % str(e))
        return None

#    try:
#        mergedsamples_single_array = np.vstack(mergedsamples)
#        fig = corner.corner(mergedsamples_single_array, quantiles=[0.16, 0.5, 0.84], plot_density=False,\
#            show_titles=True, title_fmt=".3f", labels=labels)
#        fig.savefig(output_path_plots + settings["chains_name"] + "+corner.png")
#    except Exception as e:
#        print("WARNING: %s" % str(e))

#    try:
#        return get_percentiles(mergedsamples, labels)
#    except Exception as e:
#        print("WARNING: %s" % str(e))
#        return None

def gelman_rubin_convergence(within_chain_var, mean_chain, chain_length):
    ''' Calculate Gelman & Rubin diagnostic
    # 1. Remove the first half of the current chains
    # 2. Calculate the within chain and between chain variances
    # 3. estimate your variance from the within chain and between chain variance
    # 4. Calculate the potential scale reduction parameter '''
    Nchains = within_chain_var.shape[0]
    dim = within_chain_var.shape[1]
    meanall = np.mean(mean_chain, axis=0)
    W = np.mean(within_chain_var, axis=0)
    B = np.arange(dim,dtype=np.float)
    B.fill(0)
    for jj in range(0, Nchains):
        B = B + chain_length*(meanall - mean_chain[jj])**2/(Nchains-1.)
    estvar = (1. - 1./chain_length)*W + B/chain_length
    return np.sqrt(estvar/W)

def prep_gelman_rubin(sampler):
    dim = sampler.chain.shape[2]
    chain_length = sampler.chain.shape[1]
    chainsamples = sampler.chain[:, int(chain_length/2):, :].reshape((-1, dim))
    within_chain_var = np.var(chainsamples, axis=0)
    mean_chain = np.mean(chainsamples, axis=0)
    return within_chain_var, mean_chain
