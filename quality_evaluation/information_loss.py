# -*- coding: utf-8 -*-
""" Performance Evaluator.

usage: python %s <load_img_folder> <org_extended.pickle> <org_extended.pickle of ground truth> <quality_control_method> <quality_save_folder/>
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2014/01/06"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2013 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import numpy as np
import cv
import pickle
import sys
import argparse
import datetime

#SMOOTH_VAL = 0.1

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

def _convert_result_array_to_distribution(_converted_result_array, _smoothing_parameter, _possible_labels):
    """ Convert _converted_result_array into distributions.
    To be precise, each row of _converted_result_array is converted into a distribution.
    
    :Variables:
        _converted_result_array : numpy.array
            each row contains labels given to one instance by workers.
        _smoothing_parameter : float
            a pseudo-count parameter to smooth empirical distributions.
        _possible_labels : str
            possible labels, either {binary, ten-choice}. We want to calculate a distribution on it given one instance.
    :RTypes: numpy.array
    :Returns:
        distributions. each row contains a distribution on the set of possible labels.
    """
    if _possible_labels == "binary":
        _possible_labels_list = [-1, 1]
    elif _possible_labels == "ten-choice":
        _possible_labels_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        print("ERROR: _possible_labels must be either \"binary\" or \"ten-choice\".")
        exit(-1)
    _count_array = _smoothing_parameter * np.ones((_converted_result_array.shape[0], len(_possible_labels_list)))
    _non_zero_inds = np.nonzero(_converted_result_array)
    for i in range(len(_non_zero_inds[0])):
        _count_array[_non_zero_inds[0][i], _possible_labels_list.index(_converted_result_array[_non_zero_inds[0][i], 
                                                                                               _non_zero_inds[1][i]])] += 1.
    
    _count_array = _count_array / (_count_array.sum(axis=1)[:, np.newaxis])
    return _count_array

def _align_result_array(_count_array_gt, _org_loc_list_without_repetition, _org_loc_list_without_repetition_gt):
    """ Align _count_array_gt to share the index sets with _count_array.
    
    :Variables:
        _count_array_gt : numpy.array
        _org_loc_list_without_repetition : list
        _org_loc_list_without_repetition_gt : list
            (i, (step_size * j + l * _clickable_size, step_size * k + m * _clickable_size, _clickable_size, _clickable_size))
    :RType: numpy.array
    :Returns:
        _aligned_count_array_gt
    """
    if len(_org_loc_list_without_repetition) != len(_org_loc_list_without_repetition_gt):
        print("ERROR: the lengthes of lists are different, which invades the assumption made in this program.")
        exit(-1)
    else:
        ind_list = [None] * len(_org_loc_list_without_repetition)
        for i in range(len(_org_loc_list_without_repetition)):
            ind_list[i] = _org_loc_list_without_repetition_gt.index(_org_loc_list_without_repetition[i])
            
        #print ind_list
        #print _count_array_gt[ind_list, :]
        return _count_array_gt[ind_list, :]

def _calc_information_loss(_count_array, _aligned_count_array_gt):
    """ calculate information loss using _count_array and _aligned_count_array_gt.
    
    :Variables:
        _count_array : numpy.array
        _aligned_count_array_gt : numpy.array
        *** NOTE *** 
            These two arrays must have the same index set.
    :RType: float
    :Returns:
    """
    if _count_array.shape != _aligned_count_array_gt.shape:
        print("ERROR: inconsistent shapes.")
        exit(-1)
    else:
        dist_sum = 0
        for i in range(_count_array.shape[0]):
            dist_sum += _distance(_count_array[i, :], _aligned_count_array[i, :])
        dist_sum = dist_sum / float(_count_array.shape[0])
    return dist_sum

def _distance(array1, array2, dist_type="kl"):
    """ Calculate the distance between two arrays.
    
    :Variables:
        array1 : numpy.array
            1-d numpy array. A probability distribution?
        array2 : numpy.array
            1-d numpy array. A probability distribution?
        dist_type : str
            any in supported_dist_types = ["l1", "l2", "emd", "kl"]
    :RType: float
    :Returns: A distance between array1 and array2.
    """
    supported_dist_types = ["l1", "l2", "emd", "kl"]
    if not dist_type in supported_dist_types:
        sys.stderr.write("ERROR: " + dist_type + " is not supported.\n")
        exit(0)
    
    if dist_type == supported_dist_types[0]:
        #return 0.5 * np.linalg.norm(array1, array2, 1)
        return 0.5 * np.sum(np.abs(array1 - array2))
    if dist_type == supported_dist_types[1]:
        #return 0.5 * np.linalg.norm(array1, array2, 1)
        return 0.5 * np.linalg.norm(array1 - array2)
    if dist_type == supported_dist_types[2]:
        array1_tmp = np.hstack((array1.reshape((len(array1), 1)), 0.5 * np.identity(len(array1)), np.zeros((len(array1), len(array2)))))
        array2_tmp = np.hstack((array2.reshape((len(array2), 1)), np.zeros((len(array2), len(array1))), 0.5 * np.identity(len(array2))))
        array1_cv = cv.CreateMat(array1_tmp.shape[0], array1_tmp.shape[1], cv.CV_32FC1)
        array2_cv = cv.CreateMat(array2_tmp.shape[0], array2_tmp.shape[1], cv.CV_32FC1)
        cv.Convert(cv.fromarray(array1_tmp), array1_cv)
        cv.Convert(cv.fromarray(array2_tmp), array2_cv)
        return cv.CalcEMD2(array1_cv, array2_cv, distance_type=cv.CV_DIST_L1)
    if dist_type == supported_dist_types[3]:
        array1_tmp = np.asarray(array1, dtype=np.float)
        array2_tmp = np.asarray(array2, dtype=np.float)
        return np.sum(np.where(array2_tmp != 0, array2_tmp * np.log2(array2_tmp / array1_tmp), 0))
    else:
        return None


if __name__ == "__main__":
    """
    # test for _convert_result_array_to_distribution
    _converted_result_array = np.array([[1, 0, -1], [1, 1, 1], [-1, -1, 0], [1, -1, 1]])
    _smoothing_parameter = 0.0
    _possible_labels = [-1, 1]
    _convert_result_array_to_distribution(_converted_result_array, _smoothing_parameter, _possible_labels)
    exit(0)
    """
    
    """
    # test for _align_result_array
    _count_array_gt = np.array([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [1.0, 0.0]])
    _org_loc_list_without_repetition = [(0, (0, 0, 10, 10)), (1, (10, 10, 10, 10)), (0, (0, 10, 10, 10)), (2, (20, 20, 10, 10))]
    _org_loc_list_without_repetition_gt = [(1, (10, 10, 10, 10)), (0, (0, 0, 10, 10)), (2, (20, 20, 10, 10)), (0, (0, 10, 10, 10))]
    _align_result_array(_count_array_gt, _org_loc_list_without_repetition, _org_loc_list_without_repetition_gt)
    exit(0)
    """
    
    parser = argparse.ArgumentParser(description="Calculate an information loss using converted_results.pickle files of a target task and the ground truth one.")
    parser.add_argument("converted_result", type=str, help="converted_result.pickle created by convert_data.py")
    parser.add_argument("converted_result_ground_truth", type=str, help="converted_result.pickle created by convert_data.py ** GROUND TRUTH **")
    parser.add_argument("smoothing_parameter", type=float, help="smoothing parameter to estimate empirical distributions.")
    parser.add_argument("possible_labels", type=float, help="{binary, ten-choice}.")
    parser.add_argument("save_dir", type=str, help="A directory to save clipped results and miscs.")
    args = parser.parse_args()
    
    command_date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(args)
    print("Command was executed on " + command_date)

    (org_loc_list_without_repetition, converted_result_array) = _load_pickle_files(args.converted_result)
    (org_loc_list_without_repetition_gt, converted_result_array_gt) = _load_pickle_files(args.converted_result_ground_truth)
    count_array = _convert_result_array_to_distribution(converted_result_array, args.smoothing_parameter, args.possible_labels)
    count_array_gt = _convert_result_array_to_distribution(converted_result_array_gt, args.smoothing_parameter, args.possible_labels)
    aligned_count_array_gt = _align_result_array(count_array_gt, org_loc_list_without_repetition, org_loc_list_without_repetition_gt)
    information_loss = _calc_information_loss(count_array, aligned_count_array_gt)
    print("information_loss =", information_loss)
    exit(0)
