# -*- coding: utf-8 -*-
""" Image divider.

usage: python %s <load_img_folder/> <save_folder_name> <subinstance_size> <num_subinstance_on_one_side>
REMARK:
    subinstance_size must be even.
SAVE_FILES:
    relation_between_org_and_shuffle.pkl : pickle file containing 
    org.pkl : pickle version of list 'patch_org_loc_list'
        the i-th element equals to (NAME_OF_ORIGINAL_FILE, (left-top-row, left-top-col, height, width))
    shuffled.pkl : pickle version of list 'patch_shuffled_loc_list'
        the i-th element equals to (NAME_OF_SHUFFLED_FILE, (row, column)) (NOTE: row and column are 0-origin)
    the i-th elements of both lists indicate the same object.
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2014/04/14"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2014 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

MARGIN = 0
GAP_SIZE = 0
#SIZE = 10 # finally each shuffled picture contains SIZE \times SIZE patches

import numpy as np
import cv2
import datetime
import sys
import os
import glob
import pickle
import argparse


def create_output_dir(output_dir, dir_name):
    """ create a save directory
    
    :Variables:
        output_dir : str
        dir_name : str
    """
    try:
        os.mkdir(os.path.join(output_dir, dir_name))
    except OSError:
        print(os.path.join(output_dir, dir_name) + " exits... :(")

def write_log(output_dir, texts, new_file=False):
    """ write texts into a log file in output_dir.

    :Variables:
        output_dir : str
        new : bool
            if new = True, then create a new file and delete the old file.
    """
    if new_file:
        f = open(os.path.join(output_dir, "std.log"), "w")
    else:
        f = open(os.path.join(output_dir, "std.log"), "a")
    f.write(str(texts) + "\n")
    f.close()

def load_images(load_path, normalize=True, subinstance_size=50):
    """ load images and enlarge images so that the size is propotional to `subinstance_size`. if normalize is True.

    :Variables:
        load_path : str
            load path that contains images. slash at the last.
        normalize : bool
            normalize images or not. If normalization is true, the size of each image is set to be proportional to subinstance_size.
        subinstance_size : int
            the size of a subinstance.
    :RType: list
    :Returns:
        img_list : list
            img_list[i] contains an image in a numpy.array format.
    """
    file_list = glob.glob(os.path.join(load_path, '*.jpg'))
    num_files = len(file_list)
    print("#(files) =", num_files)
    step_size = subinstance_size // 2
    img_list = [None] * num_files
    i = 0
    if normalize:
        for each_file in file_list:
            tmp_file = np.asarray(cv2.imread(each_file))
            row_size = tmp_file.shape[0]
            col_size = tmp_file.shape[1]
            if row_size <= subinstance_size:
                row_size = subinstance_size
            if col_size <= subinstance_size:
                col_size = subinstance_size
            if row_size > subinstance_size and row_size % step_size != 0:
                row_size = row_size + (step_size - row_size % step_size)
            if col_size > subinstance_size and col_size % step_size != 0:
                col_size = col_size + (step_size - col_size % step_size)
            tmp_file_1 = np.zeros((row_size, col_size, 3))
            tmp_file_1[0 : tmp_file.shape[0], 0 : tmp_file.shape[1]] = tmp_file
            img_list[i] = tmp_file_1
            i += 1
    else:
        for each_file in file_list:
            img_list[i] = np.asarray(cv2.imread(each_file))
            i += 1
    return img_list

def collect_subinstances(img_list, subinstance_size, clickable_size):
    """ create a list of subinstances from a list of NORMALIZED images.

    :Variables:
        img_list : list
            a list of images where each element is an image of a numpy.array format.
        subinstance_size : int
            the size of subinstance (step_size will be a half of it).
        clickable_size : int
            the size of a clickable area (that can divide subinstance_size well).
    :RType: tuple of lists
    :Returns:
        _subinstance_list : list
            a list of subinstances divided into clickable areas (which we call sub-subinstance).
        _subinstance_org_loc_list : list
            _subinstance_org_loc_list[i] = the original location of the i-th sub-subinstance of _subinstance_list.
    """
    num_files = len(img_list)
    step_size = subinstance_size // 2
    num_subinstances = 0
    for i in range(num_files):
        num_rows = int(np.ceil((2.0 * img_list[i].shape[0] / float(subinstance_size)) - 1.0))
        num_cols = int(np.ceil((2.0 * img_list[i].shape[1] / float(subinstance_size)) - 1.0))
        num_subinstances += num_rows * num_cols
        #print(str(i) + "-th image: " + "num_rows = " + str(num_rows) + ", num_cols = " + str(num_cols) + ", #(sub_files) = ", str(num_subinstances))
    
    print("#(subinstances) =", num_subinstances)
    
    expand = (subinstance_size // clickable_size)
    
    subinstance_list = [None] * expand * expand * num_subinstances
    subinstance_org_loc_list = [None] * expand * expand * num_subinstances
    
    patch_i = 0
    for i in range(num_files):
        num_rows = int(np.ceil((2.0 * img_list[i].shape[0] / float(subinstance_size)) - 1.0))
        num_cols = int(np.ceil((2.0 * img_list[i].shape[1] / float(subinstance_size)) - 1.0))
        for j in range(num_rows):
            for k in range(num_cols):
                for l in range(expand):
                    for m in range(expand):
                        try:
                            subinstance_list[patch_i * expand * expand + l * expand + m ] \
                                = img_list[i][step_size * j + l * clickable_size : step_size * j + (l + 1) * clickable_size,
                                              step_size * k + m * clickable_size : step_size * k + (m + 1) * clickable_size]
                        except IndexError:
                            sys.stdout.write("ERROR: images are not normalized.\n")
                            exit(0)
                        subinstance_org_loc_list[patch_i * expand * expand + l * expand + m ] \
                            = (i, (step_size * j + l * clickable_size,
                                   step_size * k + m * clickable_size,
                                   clickable_size,
                                   clickable_size))
                patch_i += 1

    if patch_i != num_subinstances:
        sys.stdout.write("ERROR: the number of subinstances is not consistent.\n")
        exit(-1)
    return subinstance_list, subinstance_org_loc_list


def combine_subinstances(output_dir, subinstance_list, subinstance_size, clickable_size, num_subinstances_to_combine, seed=42):
    """ combine subinstances to create a mosaic to crowdsource.

    :Variables:
        subinstance_list : list
        subinstance_org_loc_list : list
        subinstance_size : int
        clickable_size : int
        num_subinstances_to_combine : int
    :RType: tuple of lists
    :Returns:
        mosaic_img_list : list
            list of mosaics where each element corresponds to a mosaic in a numpy.array format.
        mosaic_loc_list : list
            mosaic_loc_list[i] = the location of the i-th sub-subinstance of subinstance_list in mosaic_img_list.
            In other words, mosaic_loc_list and subinstance_org_loc_list share the index set.
    """
    try:
        os.mkdir(os.path.join(output_dir, "mosaics"))
    except OSError:
        print(os.path.join(output_dir, "mosaics") + " exits... :(")

    expand = (subinstance_size // clickable_size)
    num_subinstances = len(subinstance_list) // (expand * expand)
    np.random.seed(seed)
    perm = np.random.permutation(num_subinstances)
    
    num_result_files = int(np.ceil(float(num_subinstances) / float(num_subinstances_to_combine * num_subinstances_to_combine)))
    print("#(mosaics) =", num_result_files)

    mosaic_img_list = [None] * num_result_files
    mosaic_loc_list = [None] * num_subinstances * expand * expand
    patch_i = 0
    for file_i in range(num_result_files):
        mosaic_img = np.zeros((subinstance_size * num_subinstances_to_combine, subinstance_size * num_subinstances_to_combine, 3))
        for i in range(num_subinstances_to_combine):
            for j in range(num_subinstances_to_combine):
                if patch_i < num_subinstances:
                    for l in range(expand):
                        for m in range(expand):
                            mosaic_img[i * subinstance_size + l * clickable_size : i * subinstance_size + (l + 1) * clickable_size,
                                       j * subinstance_size + m * clickable_size : j * subinstance_size + (m + 1) * clickable_size, :] \
                                       = subinstance_list[perm[patch_i] * expand * expand + l * expand + m]
                            mosaic_loc_list[perm[patch_i] * expand * expand + l * expand + m] \
                                = (file_i, (subinstance_size * i + l * clickable_size, 
                                            subinstance_size * j + m * clickable_size, 
                                            clickable_size, 
                                            clickable_size))
                    patch_i += 1
        cv2.imwrite(os.path.join(output_dir, "mosaics", str(file_i)+".jpg"), mosaic_img)

    return mosaic_loc_list

def save_parameters(output_dir, args, img_list, 
                    subinstance_org_loc_list, mosaic_loc_list):
    """ save parameters

    :Variables:
        args : Namespace
        output_dir : str
        img_list : list
        mosaic_img_list : list
        subinstance_org_loc_list : list
        mosaic_loc_list : list
    """
    f = open(os.path.join(output_dir, 'parameters.pkl'), 'wb')
    pickle.dump((args, img_list, subinstance_org_loc_list, mosaic_loc_list), f)
    f.close()


def main():
    parser = argparse.ArgumentParser(description="implementation of the instance clipping function.")
    parser.add_argument("input_img_dir", type=str, help="A directry of original images.")
    parser.add_argument("output_dir", type=str, help="A directory to save clipped results and miscs.")
    parser.add_argument("subinstance_size", type=int, help="The size of a clipping window [pixel].")
    parser.add_argument("clickable_size", type=int, help="The size of a clickable area [pixel].")
    parser.add_argument("num_subinstances_to_combine", type=int, help="The number of subinstances on one side of a combined image.")
    args = parser.parse_args()
    if args.subinstance_size % 2 != 0:
        print("Error: please make subinstance_size even.")
        exit(1)
    if args.subinstance_size % args.clickable_size != 0:
        print("Error: please make subinstance_size % clickable_size == 0.")
        exit(1)
    
    command_date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    create_output_dir(args.output_dir,
                      str(args.subinstance_size) + "_" + str(args.clickable_size) + "_" + str(args.num_subinstances_to_combine))
    output_path = os.path.join(args.output_dir,
                             str(args.subinstance_size) + "_" + str(args.clickable_size) + "_" + str(args.num_subinstances_to_combine))
    print(args)
    print("Command was executed on " + command_date)
    write_log(output_path, args, True)
    write_log(output_path, "Command was executed on " + command_date, False)

    img_list = load_images(args.input_img_dir,
                           normalize=True,
                           subinstance_size=args.subinstance_size)
    subinstance_list, subinstance_org_loc_list = collect_subinstances(img_list,
                                                                      args.subinstance_size,
                                                                      args.clickable_size)
    mosaic_loc_list = combine_subinstances(output_path,
                                           subinstance_list,
                                           args.subinstance_size,
                                           args.clickable_size,
                                           args.num_subinstances_to_combine)
    save_parameters(output_path, args, img_list, subinstance_org_loc_list, mosaic_loc_list)

if __name__ == "__main__":
    main()
