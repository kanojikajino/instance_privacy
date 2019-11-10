#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2014/04/14"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2014 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import pickle


def load_pickle_files(input_file_path):
    """ load parameters created by instance_clipping_and_mixing.py
    
    :Variables:
        input_file_path : str
            A path to a pickle file.
    :RType: 
    :Returns:
        loaded file
    """
    with open(input_file_path, "rb") as f:
        tmp = pickle.load(f)
    return tmp
