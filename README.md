# BAO Fitting code

Python code for fitting BAO multipoles that uses nbodykit, emcee and python functions. The code implements the Beutler et al. 2017 (Arxiv: 1607.03149) isotropic and anisotropic bao models. The original code grounds were based on [fbeutler-code](https://github.com/fbeutler/Study-the-Universe-with-Python). The structure was modified and separated into different modules.

Python > 3.5 is needed to run the code. Install a nbodykit conda environment as shown in [nbodykit-install](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html).
In my case I ran `conda create --name nbodykit-env python=3.6` using python3.6.

An example jupyter notebook for illustrating how some of the code elements work is provided. (to be completed)

## modules
- **bao_fitter.py:** Body of the code, will call different modules for calculations.
- **input.py:** Read through the input parameters and settings provided inside a .ini file in /inifiles. This module adapt the input settings from the user so the code can read them.
- **bao_template.py:** It calculates the oscillation factor for BAO based on the power spectrums produced by CLASS and a NonWiggle-Einsestein-Hu spectrum or the Hinton method.
- **bao_model_iso.py:** Calculates a power spectrum template for the isotropic analysis. (to be completed)
- **bao_model_ani.py:** Calculates a power spectrum template for the anisotropic analysis.  
- **bao_utils.py:** It contains specific functions (e.g. related with working some numpy arrays) that help some modules to run properly.
- **mcmc.py:** It does the inspection of the chains and calculate confidence limits for the bao parameters.

## folders
- data: Store input data (power spectrum, multipoles, covariance matrix, etc.)
- inifiles: Here user selects the initial parameters and settings for the BAO analysis. 
- notebooks: examples using some elements of the code into a jupyter notebook (I ran it with nbodykit through a conda environment)
