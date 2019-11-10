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
import cv2
import pickle
import sys
import os
import argparse
import datetime
from .crowd_data import BinaryData
from .lcmodel import LatentClassModel

def load_pickle_files(_file_str):
    """ load parameters created by instance_clipping_and_mixing.py
    
    :Variables:
        _file_str : str
            A path to a pickle file.
    :RType: 
    :Returns:
        loaded file
    """
    with open(_file_str, "rb") as f:
        tmp = pickle.load(f)
    return tmp


def save_parameters(parameters, save_dir, save_file_name):
    """ Save parameters into a pickle file.
    """
    with open(os.path.join(save_dir, save_file_name), "wb") as f:
        pickle.dump(parameters, f)


def aggregate_crowd_labels(crowd_res, qc_method, save_dir=None):
    """ Aggregate multiple labels on one instance to return a list of positive instances.
    
    :Variables:
        crowd_res : crowd_data.binaryData
        qc_method : str
        save_dir : str
    :RType: list
    :Returns:
        a list of positive subsubinstances
    """
    if qc_method == "no":
        sys.stdout.write("--- No Quality Control ---\n")
        pos_ind_list = list(set((crowd_res.y > 0).nonzero()[0]))
    elif qc_method == "mv":
        sys.stdout.write("--- MV Method ---\n")
        sum_y = crowd_res.y.sum(axis=1) #sum_y[i] = #(pos) - #(neg)
        pos_ind_list = []
        for i in range(len(sum_y)):
            if sum_y[i] > 0 or (sum_y[i] == 0 and np.random.binomial(1,0.5) == 0):
                pos_ind_list.append(i)
    elif qc_method == "lc":
        lc = LatentClassModel(crowd_res)
        lc.run_em(10**(-10))
        save_parameters(lc, save_dir, "lc_model.pickle")
        est_labels = lc.estimated_labels(0.5)
        pos_ind_list = np.nonzero(est_labels == 1)[0]
    else:
        sys.stdout.write("qc_method must be either {no, mv, lc}.\n")
        exit(-1)

    if save_dir is not None:
        save_parameters(pos_ind_list, save_dir, "pos_ind_list.pickle")
    return pos_ind_list

def create_masked_image(pos_ind_list, img_list, org_loc_list_without_repetition, save_dir):
    for i in pos_ind_list:
        patch_info = org_loc_list_without_repetition[i]
        # (i, (step_size * j + l * _clickable_size, step_size * k + m * _clickable_size, _clickable_size, _clickable_size))
        file_id = patch_info[0]
        try:
            img_list[file_id][patch_info[1][0] : min(patch_info[1][0] + patch_info[1][2], img_list[file_id].shape[0]),
                              patch_info[1][1] : min(patch_info[1][1] + patch_info[1][3], img_list[file_id].shape[1]),
                              :] = (255, 0, 255)
        except ValueError:
            sys.stderr.write(str(patch_info))
    try:
        os.mkdir(os.path.join(save_dir, "masked_images"))
    except OSError:
        print(os.path.join(save_dir, "masked_images"), "existed.")
    for i in range(len(img_list)):
        cv2.imwrite(os.path.join(save_dir, "masked_images", str(i) + ".jpg"), img_list[i])
    return 0

def main():
    parser = argparse.ArgumentParser(description="Draw results using converted_results.pickle and parameters.pickle.")
    parser.add_argument("parameters_file", type=str, help="parameters.pickle created by instance_clipping_and_mixing.py")
    parser.add_argument("converted_result", type=str, help="converted_result.pickle created by convert_data.py")
    parser.add_argument("quality_control", type=str, help="specify a quality control method from {no, mv, lc}.")
    parser.add_argument("save_dir", type=str, help="A directory to save clipped results and miscs.")
    args = parser.parse_args()

    command_date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(args)
    print("Command was executed on " + command_date)

    org_loc_list_without_repetition, converted_result_array = load_pickle_files(args.converted_result)
    args_ic, img_list, mosaic_img_list, subinstance_org_loc_list, mosaic_loc_list = load_pickle_files(args.parameters_file)
    crowd_res = BinaryData(converted_result_array)
    pos_ind_list = aggregate_crowd_labels(crowd_res, args.quality_control, args.save_dir)
    create_masked_image(pos_ind_list, img_list, org_loc_list_without_repetition, args.save_dir)

if __name__ == "__main__":
    main()
