# -*- coding: utf-8 -*-
""" Data Converter.

args_ic : 
    ("load_img_dir", type=str, help="A directry of original images.")
    ("save_dir", type=str, help="A directory to save clipped results and miscs.")
    ("subinstance_size", type=int, help="The size of a clipping window [pixel].")
    ("clickable_size", type=int, help="The size of a clickable area [pixel].")
    ("num_subinstances_to_combine", type=int, help="The number of subinstances on one side of a combined image.")
img_list : img_list[i] contains the i-th image.
mosaic_img_list : mosaic_img_list[i] contains the i-th mosaic.
subinstance_org_loc_list : 
    subinstance_org_loc_list[patch_i * expand * expand + l * expand + m ] 
        = (i, (step_size * j + l * _clickable_size, step_size * k + m * _clickable_size, _clickable_size, _clickable_size))
mosaic_loc_list : 
    mosaic_loc_list[perm[patch_i] * expand * expand + l * expand + m] = 
        (file_i, (_subinstance_size * i + l * _clickable_size, 
         _subinstance_size * j + m * _clickable_size, 
         _clickable_size, 
         _clickable_size))
instance_ids[0] = 'https://dl.dropboxusercontent.com/u/17481238/20131003/performance_50/htmls/0.25.html'
result_array = array([ '46.899 06_07 07_07 10_16 10_17 11_16 11_17 12_22 13_22 16_22 16_23 17_22 17_23',
                       None, None, None, None, None, None, None, None, None, None, None,
                       None, None, None, None, None, None, None, None, None, None, None,
                       None, None, None, None, None, None, None, None, None, None, None,
                       None, None, None, None, None, None, None, None, None, None, None,
                       None, None], dtype=object)
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2013/12/13"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2013 Hiroshi Kajino all rights reserved."
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

def _save_parameters(_parameters, _save_dir):
    """ Save parameters into a pickle file.
    """
    f = open(os.path.join(_save_dir, "converted_results.pickle"), "w")
    pickle.dump(_parameters, f)
    f.close()
    return 0

def _convert_to_crowdData(_subinstance_size, _clickable_size, _num_instances_to_combine, _subinstance_org_loc_list, _mosaic_loc_list,  _result_array, _instance_ids):
    """ Convert a raw result_array into crowdData where an instance is indexed by a clickable area in the original images.
    
    :Variables:
        _subinstance_size : int
            the size of a subinstance
        _clickable_size : int
            the size of a clickable area
        _num_instances_to_combine : int
            the number of subinstances on one side of a combined image.
        _subinstance_org_loc_list : list
            a list containing the place in the original image the i-th subsubinstance was.
        _mosaic_loc_list : list
            a list containing the place in the mosaic image the i-th subsubinstance was.
        _result_array : numpy.array
            result_array obtained from import-lancers-integrate-output.py
        _instance_ids : list
            the i-th row of result_array corresponds to _instance_ids[i]
    :RType: crowdData
    :Returns:
        (_org_loc_list_without_repetition, _converted_result_array)
        _org_loc_list_without_repetition : list
            A list deleting repetitions in _subinstance_Org_loc_list.
    """
    _org_loc_list_without_repetition = list(set(_subinstance_org_loc_list))
    _converted_result_array = np.zeros((len(_org_loc_list_without_repetition), _result_array.shape[1]))
    
    # create a pos_labels & neg_labels that contain the ids of positive & negative instances
    np.set_printoptions(threshold=np.nan)
    pat_space = re.compile(' ')
    pat_unsco = re.compile('_')
    pat_per = re.compile('\.')
    pos_labels = []
    neg_labels = []
    all_labels = set([]) # this is necessary in order to obtain NEGATIVE labels.
    num_pos = 0
    num_neg = 0
    num_all = 0
    for k in range(int(_num_instances_to_combine * (_subinstance_size / _clickable_size))):
        for l in range(int(_num_instances_to_combine * (_subinstance_size / _clickable_size))):
            all_labels.add((k,l))
    
    for i in range(_result_array.shape[0]):
        tmp = os.path.basename(_instance_ids[i]) # _instance_ids[i] = 'https://dl.dropboxusercontent.com/u/17481238/20130930/performance_50/htmls/99.25.html'
        mosaic_id = int(pat_per.split(tmp)[0])
        for j in range(_result_array.shape[1]):
            if _result_array[i][j] != None:
                tmp = pat_space.split(_result_array[i][j])[1:] # [0] = elapsed time, ["01_00", "02_03"]
                pos_tmp = set([]) # necessary to calculate neg_labels.
                for k in tmp:
                    fine_row = int(pat_unsco.split(k)[0])
                    fine_col = int(pat_unsco.split(k)[1])
                    pos_tmp.add((fine_row, fine_col))
                num_pos += len(pos_tmp)
                neg_tmp = list(all_labels.difference(pos_tmp))
                num_neg += len(neg_tmp)
                num_all += len(all_labels)
                
                for pos in pos_tmp:
                    try:
                        ind = _mosaic_loc_list.index((mosaic_id, (_clickable_size * pos[0], _clickable_size * pos[1], _clickable_size, _clickable_size)))
                    except ValueError:
                        print (mosaic_id, (_clickable_size * pos[0], _clickable_size * pos[1], _clickable_size, _clickable_size))
                    org_loc = subinstance_org_loc_list[ind]
                    ind_wo_repetition = _org_loc_list_without_repetition.index(org_loc)
                    _converted_result_array[ind_wo_repetition, j] = 1
                
                for neg in neg_tmp:
                    try:
                        ind = _mosaic_loc_list.index((mosaic_id, (_clickable_size * neg[0], _clickable_size * neg[1], _clickable_size, _clickable_size)))
                    except ValueError:
                        print (mosaic_id, (_clickable_size * neg[0], _clickable_size * neg[1], _clickable_size, _clickable_size))
                    org_loc = subinstance_org_loc_list[ind]
                    ind_wo_repetition = _org_loc_list_without_repetition.index(org_loc)
                    _converted_result_array[ind_wo_repetition, j] = -1
                    
    print "num_pos =", num_pos
    print "num_neg =", num_neg
    print "num_all =", num_all
    return (_org_loc_list_without_repetition, _converted_result_array)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lancers_result.pickle to more friendly data, using parameters.pickle.")
    parser.add_argument("parameters_file", type=str, help="parameters.pickle created by instance_clipping_and_mixing.py")
    parser.add_argument("lancers_result", type=str, help="lancers_result.pickle created by import-lancers-integrate-output.py")
    parser.add_argument("save_dir", type=str, help="A directory to save clipped results and miscs.")
    args = parser.parse_args()
    
    command_date = datetime.datetime.now().strftime(u'%Y/%m/%d %H:%M:%S')
    print args
    print "Command was executed on " + command_date
    
    (args_ic, img_list, mosaic_img_list, subinstance_org_loc_list, mosaic_loc_list) = _load_pickle_files(args.parameters_file)
    (worker_ids, instance_ids, result_array) = _load_pickle_files(args.lancers_result)
    (org_loc_list_without_repetition, converted_result_array) = _convert_to_crowdData(args_ic.subinstance_size, args_ic.clickable_size, args_ic.num_subinstances_to_combine, subinstance_org_loc_list, mosaic_loc_list, result_array, instance_ids)
    _save_parameters((org_loc_list_without_repetition, converted_result_array), args.save_dir)
    exit(0)
