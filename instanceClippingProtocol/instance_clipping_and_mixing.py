#!/Library/Frameworks/EPD64.framework/Versions/Current/bin/python
# -*- coding: utf-8 -*-
""" Image divider.

usage: python %s <load_img_folder/> <save_folder_name> <subinstance_size> <num_subinstance_on_one_side>
REMARK:
    subinstance_size must be even.
SAVE_FILES:
    relation_between_org_and_shuffle.pickle : pickle file containing 
    org.pickle : pickle version of list 'patch_org_loc_list'
        the i-th element equals to (NAME_OF_ORIGINAL_FILE, (left-top-row, left-top-col, height, width))
    shuffled.pickle : pickle version of list 'patch_shuffled_loc_list'
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
import cv
import cv2
import datetime
import sys
import os
import glob
import pickle
import argparse

def _create_save_dir(_save_dir, _folder_name):
    """ create a save directory
    
    :Variables:
        _save_dir : str
        _folder_name : str
    """
    try:
        os.mkdir(os.path.join(_save_dir, _folder_name))
    except OSError:
        print os.path.join(_save_dir, _folder_name) + " exits... :("
        #exit(0)
       

def _write_log(_save_path, _texts, _new):
    """ write _texts into a log file in _save_path.

    :Variables:
        _save_path : str
        _folder_name : str
        _new : bool
            if _new = True, then create a new file and delete the old file.
    """
    if _new:
        f = open(os.path.join(_save_path, "std.log"), "w")
    else:
        f = open(os.path.join(_save_path, "std.log"), "a")
    f.write(str(_texts) + "\n")
    f.close()

def _load_images(_load_path, normalize=True, _subinstance_size=50):
    """ load images
    
    :Variables:
        _load_path : str
            load path that contains images. slash at the last.
        normalize : bool
            normalize images or not. normalization means that setting the size of images proportional to _subinstance_size.
        _subinstance_size : int
            the size of a subinstance.
    :RType: list
    :Returns:
        _img_list : list
            img_list[i] contains an image in a numpy.array format.
    """
    _file_list = glob.glob(os.path.join(_load_path, '*.jpg'))
    num_files = len(_file_list)
    print "#(files) =", num_files
    step_size = _subinstance_size / 2
    _img_list = [None] * num_files
    i = 0
    if normalize:
        for file in _file_list:
            tmp_file = np.asarray(cv.LoadImageM(file))
            row_size = tmp_file.shape[0]
            col_size = tmp_file.shape[1]
            if row_size <= _subinstance_size:
                row_size = _subinstance_size
            if col_size <= _subinstance_size:
                col_size = _subinstance_size
            if row_size > _subinstance_size and row_size % step_size != 0:
                row_size = row_size + (step_size - row_size % step_size)
            if col_size > _subinstance_size and col_size % step_size != 0:
                col_size = col_size + (step_size - col_size % step_size)
            tmp_file_1 = np.zeros((row_size, col_size, 3))
            tmp_file_1[0 : tmp_file.shape[0], 0 : tmp_file.shape[1]] = tmp_file
            _img_list[i] = tmp_file_1
            i += 1
    else:
        for file in _file_list:
            _img_list[i] = np.asarray(cv.LoadImageM(file))
            i += 1
    return _img_list
    
def _collect_subinstances(_img_list, _subinstance_size, _clickable_size):
    """ create a list of subinstances from a list of NORMALIZED images.
    
    :Variables:
        _img_list : list
            a list of images where each element is an image of a numpy.array format.
        _subinstance_size : int
            the size of subinstance (step_size will be a half of it).
        _clickable_size : int
            the size of a clickable area (that can divide _subinstance_size well).
    :RType: tuple of lists
    :Returns:
        _subinstance_list : list
            a list of subinstances.
        _subinstance_org_loc_list : list
            _subinstance_org_loc_list[i] = the original location of the i-th subinstance of _subinstance_list.
    """
    num_files = len(_img_list)
    step_size = _subinstance_size / 2
    num_subinstances = 0
    for i in range(num_files):
        num_rows = int(np.ceil((2.0 * img_list[i].shape[0] / float(_subinstance_size)) - 1.0))
        num_cols = int(np.ceil((2.0 * img_list[i].shape[1] / float(_subinstance_size)) - 1.0))
        num_subinstances += num_rows * num_cols
        print str(i) + "-th image: " + "num_rows = " + str(num_rows) + ", num_cols = " + str(num_cols) + ", #(sub_files) = ", str(num_subinstances)
    
    print "#(subinstances) =", num_subinstances
    
    expand = (_subinstance_size / _clickable_size)
    
    _subinstance_list = [None] * expand * expand * num_subinstances
    _subinstance_org_loc_list = [None] * expand * expand * num_subinstances

    ########## NEEDS CHECK ########################    
    #exit(-1) #### check below
    
    patch_i = 0
    for i in range(num_files):
        num_rows = int(np.ceil((2.0 * img_list[i].shape[0] / float(_subinstance_size)) - 1.0))
        num_cols = int(np.ceil((2.0 * img_list[i].shape[1] / float(_subinstance_size)) - 1.0))
        for j in range(num_rows):
            for k in range(num_cols):
                for l in range(expand):
                    for m in range(expand):
                        try:
                            _subinstance_list[patch_i * expand * expand + l * expand + m ] = _img_list[i][step_size * j + l * _clickable_size : step_size * j + (l + 1) * _clickable_size,
                                                                                                          step_size * k + m * _clickable_size : step_size * k + (m + 1) * _clickable_size]
                            #_subinstance_list[patch_i] = _img_list[i][step_size * j : step_size * j + _subinstance_size,
                            #                                          step_size * k : step_size * k + _subinstance_size]
                        except IndexError:
                            sys.stdout.write("ERROR: images are not normalized.\n")
                            exit(0)
                        #_subinstance_org_loc_list[patch_i] = (i, (step_size * j, step_size * k, _subinstance_size, _subinstance_size))
                        _subinstance_org_loc_list[patch_i * expand * expand + l * expand + m ] = (i, (step_size * j + l * _clickable_size, step_size * k + m * _clickable_size, _clickable_size, _clickable_size))
                patch_i += 1
    
    if patch_i != num_subinstances:
        sys.stdout.write("ERROR: the number of subinstances is not consistent.\n")
        exit(0)
    return (_subinstance_list, _subinstance_org_loc_list)
    

def _combine_subinstances(_subinstance_list, _subinstance_org_loc_list, _subinstance_size, _clickable_size, _num_subinstances_to_combine):
    """ combine subinstances to create a mosaic to crowdsource.
    
    :Variables:
        _subinstance_list : list
        _subinstance_org_loc_list : list
        _subinstance_size : int
        _clickable_size : int
        _num_subinstances_to_combine : int
    :RType: tuple of lists
    :Returns:
        _mosaic_img_list : list
            list of mosaics where each element corresponds to a mosaic in a numpy.array format.
        _mosaic_loc_list : list
            _mosaic_loc_list[i] = the location of the i-th subinstance of _subinstance_list in _mosaic_img_list.
            In other words, _mosaic_loc_list and _subinstance_org_loc_list share the index set.
    """
    #exit(-1)
    ########## NEEDS CHECK ########################
    expand = (_subinstance_size / _clickable_size)
    num_subinstances = len(_subinstance_list) / (expand * expand)
    perm = np.random.permutation(num_subinstances)
    
    num_result_files = int(np.ceil(float(num_subinstances) / float(_num_subinstances_to_combine * _num_subinstances_to_combine)))
    print "#(mosaics) =", num_result_files
    
    _mosaic_img_list = [None] * num_result_files
    _mosaic_loc_list = [None] * num_subinstances * expand * expand
    patch_i = 0
    for file_i in range(num_result_files):
        _mosaic_img = np.zeros((_subinstance_size * _num_subinstances_to_combine, _subinstance_size * _num_subinstances_to_combine, 3))
        for i in range(_num_subinstances_to_combine):
            for j in range(_num_subinstances_to_combine):
                if patch_i < num_subinstances:
                    for l in range(expand):
                        for m in range(expand):
                            _mosaic_img[i * _subinstance_size + l * _clickable_size : i * _subinstance_size + (l + 1) * _clickable_size,
                                        j * _subinstance_size + m * _clickable_size : j * _subinstance_size + (m + 1) * _clickable_size, :] = _subinstance_list[perm[patch_i] * expand * expand + l * expand + m]
                            _mosaic_loc_list[perm[patch_i] * expand * expand + l * expand + m] = (file_i, (_subinstance_size * i + l * _clickable_size, 
                                                                                                           _subinstance_size * j + m * _clickable_size, 
                                                                                                           _clickable_size, 
                                                                                                           _clickable_size))
                    patch_i += 1
        _mosaic_img_list[file_i] = _mosaic_img

    return (_mosaic_img_list, _mosaic_loc_list)

def _combine_subinstances_backup(_subinstance_list, _subinstance_org_loc_list, _subinstance_size, _num_subinstances_to_combine):
    """ combine subinstances to create a mosaic to crowdsource.
    
    :Variables:
        _subinstance_list : list
        _subinstance_org_loc_list : list
        _subinstance_size : int
        _num_subinstances_to_combine : int
    :RType: tuple of lists
    :Returns:
        _mosaic_img_list : list
            list of mosaics where each element corresponds to a mosaic in a numpy.array format.
        _mosaic_loc_list : list
            _mosaic_loc_list[i] = the location of the i-th subinstance of _subinstance_list in _mosaic_img_list.
            In other words, _mosaic_loc_list and _subinstance_org_loc_list share the index set.
    """
    num_subinstances = len(_subinstance_list)
    perm = np.random.permutation(num_subinstances)
    
    num_result_files = int(np.ceil(float(num_subinstances) / float(_num_subinstances_to_combine * _num_subinstances_to_combine)))
    print "#(mosaics) =", num_result_files
    
    _mosaic_img_list = [None] * num_result_files
    _mosaic_loc_list = [None] * num_subinstances
    patch_i = 0
    for file_i in range(num_result_files):
        _mosaic_img = np.zeros((_subinstance_size * _num_subinstances_to_combine, _subinstance_size * _num_subinstances_to_combine, 3))
        for i in range(_num_subinstances_to_combine):
            for j in range(_num_subinstances_to_combine):
                if patch_i < num_subinstances:
                    _mosaic_img[i * _subinstance_size : i * _subinstance_size + _subinstance_size,
                                j * _subinstance_size : j * _subinstance_size + _subinstance_size, :] = _subinstance_list[perm[patch_i]]
                    _mosaic_loc_list[perm[patch_i]] = (file_i, (_subinstance_size * i, 
                                                                _subinstance_size * j, 
                                                                _subinstance_size, 
                                                                _subinstance_size))
                    patch_i += 1
        _mosaic_img_list[file_i] = _mosaic_img

    return (_mosaic_img_list, _mosaic_loc_list)


def _save_mosaics(_save_dir, _mosaic_img_list):
    """ save mosaics
    
    :Variables:
        _save_dir : str
        _mosaic_img_list : list
    """
    try:
        os.mkdir(os.path.join(_save_dir, "mosaics"))
    except OSError:
        print os.path.join(_save_dir, "mosaics") + " exits... :("
        
    for i in range(len(_mosaic_img_list)):
        cv.SaveImage(os.path.join(_save_dir, "mosaics", str(i)+".jpg"), cv.fromarray(_mosaic_img_list[i]))
    
    return 0

def _save_parameters(_save_dir, _img_list, _mosaic_img_list, _subinstance_org_loc_list, _mosaic_loc_list):
    """ save parameters
    
    :Variables:
        _save_dir : str
        _img_list : list
        _mosaic_img_list : list
        _subinstance_org_loc_list : list
        _mosaic_loc_list : list
    """
    f = open(os.path.join(_save_dir, 'parameters.pickle'), 'w')
    pickle.dump((_img_list, _mosaic_img_list, _subinstance_org_loc_list, _mosaic_loc_list), f)
    f.close()
    return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="implementation of the instance clipping function.")
    parser.add_argument("load_img_dir", type=str, help="A directry of original images.")
    parser.add_argument("save_dir", type=str, help="A directory to save clipped results and miscs.")
    parser.add_argument("subinstance_size", type=int, help="The size of a clipping window [pixel].")
    parser.add_argument("clickable_size", type=int, help="The size of a clickable area [pixel].")
    parser.add_argument("num_subinstances_to_combine", type=int, help="The number of subinstances on one side of a combined image.")
    args = parser.parse_args()
    if args.subinstance_size % 2 != 0:
        print "Error: please make subinstance_size even."
        exit(1)
    if args.subinstance_size % args.clickable_size != 0:
        print "Error: please make subinstance_size % clickable_size == 0."
        exit(1)
    
    command_date = datetime.datetime.now().strftime(u'%Y/%m/%d %H:%M:%S')
    _create_save_dir(args.save_dir, str(args.subinstance_size) + "_" + str(args.num_subinstances_to_combine))
    save_path = os.path.join(args.save_dir, str(args.subinstance_size) + "_" + str(args.num_subinstances_to_combine))
    print args
    print "Command was executed on " + command_date
    _write_log(save_path, args, True)
    _write_log(save_path, "Command was executed on " + command_date, False)
    
    img_list = _load_images(args.load_img_dir, normalize=True, _subinstance_size=args.subinstance_size)
    (subinstance_list, subinstance_org_loc_list) = _collect_subinstances(img_list, args.subinstance_size, args.clickable_size)
    (mosaic_img_list, mosaic_loc_list) = _combine_subinstances(subinstance_list, subinstance_org_loc_list, args.subinstance_size, args.clickable_size, args.num_subinstances_to_combine)
    _save_mosaics(save_path, mosaic_img_list)
    _save_parameters(save_path, img_list, mosaic_img_list, subinstance_org_loc_list, mosaic_loc_list)
