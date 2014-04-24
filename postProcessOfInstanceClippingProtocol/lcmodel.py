# -*- coding: utf-8 -*-
""" Binary latent class model.
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2013/12/15"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2013 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import numpy as np
import scipy as sp
import scipy.misc
import sys
import crowdData

DEBUG=1

class lcModel:
    """ Binary latent class model
    
    :IVariables:
        data : crowdData.binaryData
        log_mu : numpy.array
            numpy.array of length I. mu = Pr[true_label = 1 | other variables].
        log_p : numpy.array
            numpy.array of length 2. p = Pr[true_label = 1]. The 1st element contains log(p), and the 2nd log(1-p).
        log_a : numpy.array
            numpy.array of length I. intermediate variables to calculate log_p.
        log_b : numpy.array
            numpy.array of length I. intermediate variables to calculate log_p.
        log_alpha : numpy.array
            2 * J numpy.array. alpha_j = Pr[label_by_worker_j = 1 | true_label = 1]. The 1st row contains log(aloha), and the 2nd log(1-alpha).
        log_beta : numpy.array
            numpy.array of length J. beta_j = Pr[label_by_worker_j = 0 | true_label = 0]. The 1st row contains log(beta), and the 2nd log(1-beta).
    """
    def __init__(self, _crowdData):
        self.data = _crowdData
        self.pos_ind = np.where(self.data.y == 1)
        self.neg_ind = np.where(self.data.y == -1)
        self.log_mu = self.data.majority_vote("log_prob")
        self.log_p = np.zeros(2)
        self.log_a = np.zeros(self.data.num_instance())
        self.log_b = np.zeros(self.data.num_instance())
        self.log_alpha = np.zeros((2, self.data.num_worker()))
        self.log_beta = np.zeros((2, self.data.num_worker()))
        self._m_step()

    def _e_step(self):
        """ Perform the E-step. I.e., update log_a, log_b, and log_mu.
        """
        self.log_a = np.sum((self.log_alpha[0, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == 1), axis = 1)\
                     + np.sum((self.log_alpha[1, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == -1), axis = 1)
        self.log_b = np.sum((self.log_beta[0, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == -1), axis = 1)\
                     + np.sum((self.log_beta[1, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == 1), axis = 1)
        self.log_mu[0, :] = self.log_p[0] + self.log_a
        self.log_mu[1, :] = self.log_p[1] + self.log_b
        self.log_mu = self.log_mu - sp.misc.logsumexp(self.log_mu, axis=0)
        return 0
        
    def _m_step(self):
        """ Perform the M-step. I.e., update log_p, log_alpha, and log_beta.
        """
        log_mu_IJ = ((self.log_mu[0,:] * np.ones((self.data.num_worker(), self.data.num_instance()))).transpose())
        log_one_minus_mu_IJ = ((self.log_mu[1,:] * np.ones((self.data.num_worker(), self.data.num_instance()))).transpose())
        alpha_log_denomi = sp.misc.logsumexp(log_mu_IJ, axis=0, b=(self.data.y != 0))
        alpha_log_nume_pos = sp.misc.logsumexp(log_mu_IJ, axis=0, b=(self.data.y == 1))
        alpha_log_nume_neg = sp.misc.logsumexp(log_mu_IJ, axis=0, b=(self.data.y == -1))
        beta_log_denomi = sp.misc.logsumexp(log_one_minus_mu_IJ, axis=0, b=(self.data.y != 0))
        beta_log_nume_pos = sp.misc.logsumexp(log_one_minus_mu_IJ, axis=0, b=(self.data.y == 1))
        beta_log_nume_neg = sp.misc.logsumexp(log_one_minus_mu_IJ, axis=0, b=(self.data.y == -1))
        self.log_p = sp.misc.logsumexp(self.log_mu, axis=1)
        self.log_p = self.log_p - sp.misc.logsumexp(self.log_p)
        self.log_alpha = np.array([alpha_log_nume_pos - alpha_log_denomi, alpha_log_nume_neg - alpha_log_denomi])
        self.log_beta = np.array([beta_log_nume_neg - beta_log_denomi, beta_log_nume_pos - beta_log_denomi])
        return 0
    
    def _q_function(self):
        """ Calculate the value of the Q-function on current estimates.
        """
        self.log_a = np.sum((self.log_alpha[0, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == 1), axis = 1)\
                     + np.sum((self.log_alpha[1, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == -1), axis = 1)
        self.log_b = np.sum((self.log_beta[0, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == -1), axis = 1)\
                     + np.sum((self.log_beta[1, :] * np.ones((self.data.num_instance(), self.data.num_worker()))) * (self.data.y == 1), axis = 1)
        log_pa_pb = np.array([self.log_p[0] + (self.log_a), self.log_p[1] + (self.log_b)])
        return (np.exp(self.log_mu) * log_pa_pb).sum()

    def run_em(self, _eps, _prnt="no"):
        """ Run EM algorithm
        
        :Variables:
           _eps : float
              tolerable relative errors on the Q-function.
           _prnt : str
              if _prnt == "yes", print the value of the q-function, else don't print.
        """
        q_new = -np.inf
        q_old = 0
        convergent = False
        while not convergent:
            q_old = q_new
            self._e_step()
            self._m_step()
            q_new = self._q_function()
            convergent = (np.abs(q_old - q_new) / np.abs(q_new) < _eps)
            if _prnt == "yes":
                if q_new - q_old < 0:
                    sys.stderr.write("WARNING: Q-function decreases. Something might be wrong.\n")
                    sys.stderr.flush()
                sys.stdout.write("\r " + "q_func = " + str(q_new) + "\n")
                sys.stdout.flush()

        sys.stdout.write("\n"+"Converged. Relative_err = " + str(np.abs(q_old - q_new) / np.abs(q_new)) + "\n")
        return 0
        
    def estimated_labels(self, threshold=0.5):
        """ Estimate the true labels based on the current estimates on the posterior probabilities of the true labels.

        :Variables:
            threshold : float
            A threshold to round the probability. If mu > threshold, return 1. Otherwise, return -1.
        :RType: numpy.array
        :Returns: Estimated labels. The length of returned numpy.array = #(instances)
        """
        return (self.log_mu[0, :] > np.log(threshold)).astype(int) * 2 - 1

if __name__ == "__main__":
    # for test
    #mat = np.array([[1,1,1,1,1,-1,-1], [-1,-1,-1,-1,-1,1,1], [1,1,1,-1,-1,-1,-1]])
    mat = np.array([[1,1,1,1,1,1,-1], [-1,-1,-1,-1,-1,-1,1], [1,1,1,1,1,1,1],[1,1,1,1,1,1,-1],[1,1,1,1,1,1,-1]])
    c_data = crowdData.binaryData(mat)
    lcTest = lcModel(c_data)
    lcTest.run_em(10**(-10))
    print lcTest.estimated_labels()
    print np.exp(lcTest.log_alpha[0,:])
    print np.exp(lcTest.log_beta[0,:])
