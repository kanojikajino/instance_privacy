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
import argparse
import datetime
from .utils import load_pickle_files

#SMOOTH_VAL = 0.1


def convert_result_array_to_distribution(converted_result_array, smoothing_parameter, possible_labels):
    """ Convert converted_result_array into distributions.
    To be precise, each row of converted_result_array is converted into a distribution.
    
    :Variables:
        converted_result_array : numpy.array
            each row contains labels given to one instance by workers.
        smoothing_parameter : float
            a pseudo-count parameter to smooth empirical distributions.
        possible_labels : str
            possible labels, either {binary, ten-choice}. We want to calculate a distribution on it given one instance.
    :RTypes: numpy.array
    :Returns:
        distributions. each row contains a distribution on the set of possible labels.
    """
    if possible_labels == "binary":
        possible_labels_list = [-1, 1]
    elif possible_labels == "ten-choice":
        possible_labels_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        raise ValueError("ERROR: possible_labels must be either \"binary\" or \"ten-choice\".")

    count_array = smoothing_parameter * np.ones((converted_result_array.shape[0], len(possible_labels_list)))
    non_zero_inds = np.nonzero(converted_result_array)
    for i in range(len(non_zero_inds[0])):
        count_array[non_zero_inds[0][i],
                    possible_labels_list.index(converted_result_array[non_zero_inds[0][i],
                                                                      non_zero_inds[1][i]])] += 1.

    count_array = count_array / (count_array.sum(axis=1)[:, np.newaxis])
    return count_array

def align_result_array(count_array_gt, org_loc_list_without_repetition, org_loc_list_without_repetition_gt):
    """ Align count_array_gt to share the index sets with count_array.

    :Variables:
        count_array_gt : numpy.array
        org_loc_list_without_repetition : list
        org_loc_list_without_repetition_gt : list
            (i, (step_size * j + l * _clickable_size, step_size * k + m * _clickable_size, _clickable_size, _clickable_size))
    :RType: numpy.array
    :Returns:
        aligned_count_array_gt
    """
    if len(org_loc_list_without_repetition) != len(org_loc_list_without_repetition_gt):
        raise ValueError("ERROR: the lengthes of lists are different, which invades the assumption made in this program.")
    else:
        ind_list = [None] * len(org_loc_list_without_repetition)
        for i in range(len(org_loc_list_without_repetition)):
            ind_list[i] = org_loc_list_without_repetition_gt.index(org_loc_list_without_repetition[i])

        return count_array_gt[ind_list, :]

def calc_information_loss(count_array, aligned_count_array_gt):
    """ calculate information loss using count_array and aligned_count_array_gt.
    
    :Variables:
        count_array : numpy.array
        aligned_count_array_gt : numpy.array
        *** NOTE *** 
            These two arrays must have the same index set.
    :RType: float
    :Returns:
    """
    if count_array.shape != aligned_count_array_gt.shape:
        raise ValueError("ERROR: inconsistent shapes.")

    dist_sum = 0
    for i in range(count_array.shape[0]):
        dist_sum += _distance(count_array[i, :], aligned_count_array_gt[i, :])
    dist_sum = dist_sum / float(count_array.shape[0])
    return dist_sum


def _distance(array1, array2, dist_type="kl"):
    """ Calculate the distance between two arrays.

    :Variables:
        array1 : numpy.array
            1-d numpy array. A probability distribution?
        array2 : numpy.array
            1-d numpy array. A probability distribution?
        dist_type : str
            any in supported_dist_types = ["l1", "l2", "kl"]
    :RType: float
    :Returns: A distance between array1 and array2.
    """
    supported_dist_types = ["l1", "l2", "kl"]
    if dist_type not in supported_dist_types:
        raise ValueError("ERROR: " + dist_type + " is not supported.\n")

    if dist_type == supported_dist_types[0]:
        return 0.5 * np.sum(np.abs(array1 - array2))
    elif dist_type == supported_dist_types[1]:
        return 0.5 * np.linalg.norm(array1 - array2)
    elif dist_type == supported_dist_types[2]:
        array1_tmp = np.asarray(array1, dtype=np.float)
        array2_tmp = np.asarray(array2, dtype=np.float)
        return np.sum(np.where(array2_tmp != 0, array2_tmp * np.log2(array2_tmp / array1_tmp), 0))
    else:
        return None

def main():
    """
    # test for convert_result_array_to_distribution
    converted_result_array = np.array([[1, 0, -1], [1, 1, 1], [-1, -1, 0], [1, -1, 1]])
    smoothing_parameter = 0.0
    possible_labels = [-1, 1]
    convert_result_array_to_distribution(converted_result_array, smoothing_parameter, possible_labels)
    exit(0)
    """
    
    """
    # test for align_result_array
    count_array_gt = np.array([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [1.0, 0.0]])
    org_loc_list_without_repetition = [(0, (0, 0, 10, 10)), (1, (10, 10, 10, 10)), (0, (0, 10, 10, 10)), (2, (20, 20, 10, 10))]
    org_loc_list_without_repetition_gt = [(1, (10, 10, 10, 10)), (0, (0, 0, 10, 10)), (2, (20, 20, 10, 10)), (0, (0, 10, 10, 10))]
    align_result_array(count_array_gt, org_loc_list_without_repetition, org_loc_list_without_repetition_gt)
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

    (org_loc_list_without_repetition, converted_result_array) = load_pickle_files(args.converted_result)
    (org_loc_list_without_repetition_gt, converted_result_array_gt) = load_pickle_files(args.converted_result_ground_truth)
    count_array = convert_result_array_to_distribution(converted_result_array, args.smoothing_parameter, args.possible_labels)
    count_array_gt = convert_result_array_to_distribution(converted_result_array_gt, args.smoothing_parameter, args.possible_labels)
    aligned_count_array_gt = align_result_array(count_array_gt, org_loc_list_without_repetition, org_loc_list_without_repetition_gt)
    information_loss = calc_information_loss(count_array, aligned_count_array_gt)
    print("information_loss =", information_loss)


if __name__ == "__main__":
    main()
