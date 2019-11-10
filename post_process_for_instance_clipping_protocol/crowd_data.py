# -*- coding: utf-8 -*-
""" Binary data class for crowdsourced training data.
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2013/12/15"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2013 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import sys

DEBUG=1

class binaryData:
    """ Binary crowd data class without feature vectors.
    
    :IVariables:
        I : int
            the number of instances.
        J : int
            the number of workers.
        y : numpy.array
            I * J numpy.array. Each column corresponds to each worker's label.
            y[i,j] == 0 if worker j doesn't label data i.
            y[i,j] == 1, or -1 if worker j labels data i.
    """
    def __init__(self, _y):
        """ Initialization
        """
        self.y = _y
        self.I = _y.shape[0]
        self.J = _y.shape[1]
        
    def num_instance(self):
        """ Return the number of instances I.
        
        :RType: int
        :Returns: the number of data.
        """
        return self.I
    
    def num_worker(self):
        """  Return the number of workers J.
        
        :RType: int
        :Returns: the number of workers.
        """
        return self.J

    def majority_vote(self, prob):
        """ Return the (soft/hard) majority votes
        
        :Variables:
            prob : str
                if prob = "prob", then return probabilities of positive labels.
                if prob = "log_prob", then return log probabilities of positive labels.
                if prob = "no", then return the majority voted labels.
        :RType: numpy.array
        :Returns: 1-d numpy.array of length I, each element contains (soft/hard) majority votes.
        """
        pos_minus_neg = self.y.sum(axis=1)
        pos_plus_neg = (self.y * self.y).sum(axis=1)
        pos_array = (pos_plus_neg + pos_minus_neg) / 2.0
        neg_array = (pos_plus_neg - pos_minus_neg) / 2.0
        if DEBUG == 1:
            if pos_minus_neg.shape != (self.I,) or pos_minus_neg.shape != (self.I,):
                sys.stderr.write("ERROR: wrong shape")
                exit(0)
            if (not np.array_equal(pos_array + neg_array, pos_plus_neg)) or (not np.array_equal(pos_array - neg_array, pos_minus_neg)):
                sys.stderr.write("ERROR: wrong results")
                exit(0)
        
        if prob == "prob":
            return pos_array.astype('float') / pos_plus_neg.astype('float')
        elif prob == "log_prob":
            return np.array([np.ma.log(pos_array) - np.ma.log(pos_plus_neg), np.ma.log(neg_array) - np.ma.log(pos_plus_neg)])
        elif prob == "no":
            sys.stderr.write("ERROR: sorry, not implemented.")
            exit(0)
            #return pos_array.astype('float') / pos_plus_neg.astype('float')

if __name__ == "__main__":
    # for test
    mat = np.array([[1,1,1,0,0,-1,-1], [1,1,1,-1,1,1,1], [-1,-1,1,0,1,-1,1]])
    c_data = binaryData(mat)
    
    print c_data.num_instance()
    print c_data.num_worker()
    #print c_data.majority_vote("log_prob") == np.log(np.array([3.0/5.0, 6.0/7.0, 3.0/6.0]))
    print c_data.majority_vote("log_prob")
    print np.log(np.array([3.0/5.0, 6.0/7.0, 3.0/6.0]))
    
