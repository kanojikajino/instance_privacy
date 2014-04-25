# -*- coding: utf-8 -*-
""" Results drawer using converted data.
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2014/04/25"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2014 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import numpy as np
import cv
import pickle
import sys
import os
import glob
import re
import os.path
import argparse
import datetime
import crowdData
import lcmodel

def _load_pickle_files(_file_str):
    """ load parameters created by instance_clipping_and_mixing.py
    
    :Variables:
        _file_str : str
            A path to a pickle file.
    :RType: 
    :Returns:
        loaded file
    """
    f = open(_file_str, "r")
    tmp = pickle.load(f)
    f.close()
    return tmp

def _save_parameters(_parameters, _save_dir, _save_file_name):
    """ Save parameters into a pickle file.
    """
    f = open(os.path.join(_save_dir, _save_file_name), "w")
    pickle.dump(_parameters, f)
    f.close()
    return 0


def _aggregate_crowd_labels(_crowd_res, _qc_method, _save_dir=None):
    """ Aggregate multiple labels on one instance to return a list of positive instances.
    
    :Variables:
        _crowd_res : crowdData.binaryData
        _qc_method : str
        _save_dir : str
    :RType: list
    :Returns:
        a list of positive subsubinstances
    """
    if _qc_method == "no":
        sys.stdout.write("--- No Quality Control ---\n")
        _pos_ind_list = list(set((_crowd_res.y > 0).nonzero()[0]))
        
    elif _qc_method == "mv":
        sys.stdout.write("--- MV Method ---\n")
        _sum_y = _crowd_res.y.sum(axis=1) #_sum_y[i] = #(pos) - #(neg)
        _pos_ind_list = []
        for i in range(len(_sum_y)):
            if _sum_y[i] > 0 or (_sum_y[i] == 0 and np.random.binomial(1,0.5) == 0):
                _pos_ind_list.append(i)
        
    elif _qc_method == "lc":
        _lc = lcmodel.lcModel(_crowd_res)
        _lc.run_em(10**(-10))
        _save_parameters(_lc, _save_dir, "lc_model.pickle")
        _est_labels = _lc.estimated_labels(0.5)
        _pos_ind_list = np.nonzero(_est_labels == 1)[0]
        
    else:    
        sys.stdout.write("_qc_method must be either {no, mv, lc}.\n")
        exit(0)
    
    if _save_dir != None:
        _save_parameters(_pos_ind_list, _save_dir, "pos_ind_list.pickle")
    return _pos_ind_list

def _create_masked_image(_pos_ind_list, _img_list, _org_loc_list_without_repetition, _save_dir):
    for i in pos_ind_list:
        patch_info = _org_loc_list_without_repetition[i] # (i, (step_size * j + l * _clickable_size, step_size * k + m * _clickable_size, _clickable_size, _clickable_size))
        file_id = patch_info[0]
        try:
            _img_list[file_id][patch_info[1][0] : min(patch_info[1][0] + patch_info[1][2], _img_list[file_id].shape[0]),
                               patch_info[1][1] : min(patch_info[1][1] + patch_info[1][3], _img_list[file_id].shape[1]), 
                               :] = (255,0,255) 
        except ValueError:
            sys.stderr.write(str(patch_info))
    try:
        os.mkdir(os.path.join(_save_dir, "masked_images"))
    except OSError:
        print os.path.join(_save_dir, "masked_images"), "existed."
    for i in range(len(_img_list)):
        cv.SaveImage(os.path.join(_save_dir, "masked_images", str(i) + ".jpg"), cv.fromarray(_img_list[i]))
        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw results using converted_results.pickle and parameters.pickle.")
    parser.add_argument("parameters_file", type=str, help="parameters.pickle created by instance_clipping_and_mixing.py")
    parser.add_argument("converted_result", type=str, help="lancers_result.pickle created by import-lancers-integrate-output.py")
    parser.add_argument("quality_control", type=str, help="specify a quality control method from {no, mv, lc}.")
    parser.add_argument("save_dir", type=str, help="A directory to save clipped results and miscs.")
    args = parser.parse_args()
    
    command_date = datetime.datetime.now().strftime(u'%Y/%m/%d %H:%M:%S')
    print args
    print "Command was executed on " + command_date
    
    (org_loc_list_without_repetition, converted_result_array) = _load_pickle_files(args.converted_result)
    (args_ic, img_list, mosaic_img_list, subinstance_org_loc_list, mosaic_loc_list) = _load_pickle_files(args.parameters_file)
    crowd_res = crowdData.binaryData(converted_result_array)
    pos_ind_list = _aggregate_crowd_labels(crowd_res, args.quality_control, args.save_dir)
    _create_masked_image(pos_ind_list, img_list, org_loc_list_without_repetition, args.save_dir)
    
    exit(0)
