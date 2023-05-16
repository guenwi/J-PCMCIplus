from __future__ import print_function
import tigramite
from tigramite.independence_tests.parcorr_mult import ParCorrMult
from tigramite.independence_tests.gpdc import GaussProcReg, GPDC
import numpy as np
import warnings
from copy import deepcopy


import json, warnings, os, pathlib
import numpy as np
import dcor
from sklearn import gaussian_process

# make adaption in ParCorr-Mult of checking for constant rows and not including the result
class GaussProcRegNew(GaussProcReg):
    def __init__(self, **kwargs):

        GaussProcReg.__init__(self, **kwargs)
        
    def _get_single_residuals(self, array, target_var,
                              return_means=False,
                              standardize=True,
                              return_likelihood=False):
        """Returns residuals of Gaussian process regression.
        Performs a GP regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated mean and the likelihood.
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns
        target_var : {0, 1}
            Variable to regress out conditions from.
        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand.
        return_means : bool, optional (default: False)
            Whether to return the estimated regression line.
        return_likelihood : bool, optional (default: False)
            Whether to return the log_marginal_likelihood of the fitted GP
        Returns
        -------
        resid [, mean, likelihood] : array-like
            The residual of the regression and optionally the estimated mean
            and/or the likelihood.
        """
        array_copy = deepcopy(array)
        # remove the parts of the array within dummy that are constant zero (ones are cut off)
        mask = np.all(array_copy == array_copy[:, 0, None], axis=1)
        
        array = array_copy[~mask]

        dim, T = array.shape
        
        if self.gp_params is None:
            self.gp_params = {}

        if dim <= 2:
            if return_likelihood:
                return array[target_var, :], -np.inf
            return array[target_var, :]

        # Standardize
        if standardize:
            #print(array, array.mean(), np.any(np.isinf(array)))
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.):
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        target_series = array[target_var, :]
        z = np.fastCopyAndTranspose(array[2:])
        if np.ndim(z) == 1:
            z = z.reshape(-1, 1)


        # Overwrite default kernel and alpha values
        params = self.gp_params.copy()
        if 'kernel' not in list(self.gp_params):
            kernel = gaussian_process.kernels.RBF() +\
             gaussian_process.kernels.WhiteKernel()
        else:
            kernel = self.gp_params['kernel']
            del params['kernel']

        if 'alpha' not in list(self.gp_params):
            alpha = 0.
        else:
            alpha = self.gp_params['alpha']
            del params['alpha']

        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                               alpha=alpha,
                                               **params)

        gp.fit(z, target_series.reshape(-1, 1))

        if self.verbosity > 3:
            print(kernel, alpha, gp.kernel_, gp.alpha)

        if return_likelihood:
            likelihood = gp.log_marginal_likelihood()

        mean = gp.predict(z).squeeze()

        resid = target_series - mean

        if return_means and not return_likelihood:
            return (resid, mean)
        elif return_likelihood and not return_means:
            return (resid, likelihood)
        elif return_means and return_likelihood:
            return resid, mean, likelihood
        return resid
    
class GPDCNew(GPDC):
    def __init__(self, **kwargs):
        
        GPDC.__init__(self,**kwargs)
        self.gauss_pr = GaussProcRegNew(null_samples= self.sig_samples,
                                     cond_ind_test=self,
                                     verbosity=self.verbosity)
        
