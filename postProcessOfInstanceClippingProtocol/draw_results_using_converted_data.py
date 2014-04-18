# -*- coding: utf-8 -*-
""" Results drawer using converted data.

usage: python %s <load_img_folder> <org_extended.pickle> <org_extended.pickle of ground truth> <quality_control_method> <img_save_folder/> <quality_save_folder/>
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2013/12/21"
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
import crowdData
import lcmodel

def distance(array1, array2, dist_type="kl"):
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
        return np.sum(np.where(array2_tmp != 0, array2_tmp * np.log(array2_tmp / array1_tmp), 0))
    return -1


if __name__ == "__main__":
    argvs = sys.argv
    argc = len(argvs)
    if(argc != 8):
        print "usage: python %s <load_img_folder> <org_extended.pickle> <org_extended.pickle of ground truth> <quality_control_method> <eval_method> <img_save_folder/> <quality_save_folder/>" % argvs[0]
        exit(0)
    else:
        """
        org_extended.pickle = [org_extended, org_res_mat]
        org_extended[0] = ('./cooking_001.jpg', (0, 0, 25, 25))
        org_res_mat : len(org_extended) * #(workers) numpy.array. each element = {+1, -1, 0}. +1 = pos_label, -1 = neg_label, 0 = not labeled.
        """
        qual_method = argvs[4]
        eval_method = argvs[5]
        img_save_folder = argvs[6]
        eval_save_folder = argvs[7]
        
        if qual_method != "LC" and qual_method != "MV" and qual_method != "No":
            sys.stderr.write("ERROR: quality_control_method must be either LC, MV, or No.")
            exit(0)
        
        if eval_method != "precision_recall" and eval_method != "information_loss":
            sys.stderr.write("ERROR: eval_method must be either precision_recall or information_loss.")
            exit(0)
        
        b_dir = os.getcwd() + '/'
        f_org = open(b_dir + argvs[2])
        [org_extended, org_res_mat]= pickle.load(f_org)
        f_org.close()
        
        f_org_ground_truth = open(b_dir + argvs[3])
        [org_extended_ground_truth, org_res_mat_ground_truth]= pickle.load(f_org_ground_truth)
        f_org_ground_truth.close()
        
        # arrange each instance in inst & extract the size of a clickalable area.
        clickalable_area_size = org_extended[0][1][2] # 25
        print "clickalable_area_size =", clickalable_area_size
        
        # load all images
        os.chdir(b_dir + argvs[1])
        file_list = glob.glob('./*.jpg')
        num_files = len(file_list)
        print "#(files) =", num_files
        img_list = [None] * num_files
        img_mask_list = [None] * num_files
        img_mask_list_ground_truth = [None] * num_files
        i = 0
        
        for file_name in file_list:
            img_list[i] = cv.LoadImageM(file_name)
            img_mask_list[i] = np.zeros((img_list[i].rows, img_list[i].cols))
            img_mask_list_ground_truth[i] = np.zeros((img_list[i].rows, img_list[i].cols))
            file_list[i] = file_name
            i += 1
        
        os.chdir(b_dir)
        
        # create masked images
        if qual_method == "No":
            sys.stdout.write("--- No Quality Control ---\n")
            pos_ind_list = list(set((org_res_mat > 0).nonzero()[0]))
            for i in pos_ind_list:
                patch_info = org_extended[i] # ('./cooking_001.jpg', (0, 0, 25, 25))
                file_id = file_list.index(patch_info[0])
                #if patch_info[1][0] + patch_info[1][2] < img_mask_list[file_id].shape[0] and patch_info[1][1] + patch_info[1][3] < img_mask_list[file_id].shape[1]:
                try:
                    img_mask_list[file_id][patch_info[1][0] : min(patch_info[1][0] + patch_info[1][2], img_mask_list[file_id].shape[0]),
                                           patch_info[1][1] : min(patch_info[1][1] + patch_info[1][3], img_mask_list[file_id].shape[1])] \
                                           += np.ones((min(patch_info[1][2], img_mask_list[file_id].shape[0] - patch_info[1][0]), min(patch_info[1][3], img_mask_list[file_id].shape[1] - patch_info[1][1])))
                except ValueError:
                    sys.stderr.write(str(patch_info))
        
        elif qual_method == "MV":
            sys.stdout.write("--- MV Method ---\n")
            pos_list = org_res_mat.sum(axis=1) #pos_list[i] = #(pos) - #(neg)
            for i in range(len(pos_list)):
                if pos_list[i] > 0 or (pos_list[i] == 0 and np.random.binomial(1,0.5) == 0):
                    patch_info = org_extended[i] # ('./cooking_001.jpg', (0, 0, 25, 25))
                    file_id = file_list.index(patch_info[0])
                    try:
                        img_mask_list[file_id][patch_info[1][0] : min(patch_info[1][0] + patch_info[1][2], img_mask_list[file_id].shape[0]),
                                               patch_info[1][1] : min(patch_info[1][1] + patch_info[1][3], img_mask_list[file_id].shape[1])] \
                                               += np.ones((min(patch_info[1][2], img_mask_list[file_id].shape[0] - patch_info[1][0]), min(patch_info[1][3], img_mask_list[file_id].shape[1] - patch_info[1][1])))
                    except ValueError:
                        sys.stderr.write(str(patch_info))
                        
        elif qual_method == "LC":
            sys.stdout.write("--- LC Method ---\n")
            crowd_res = crowdData.binaryData(org_res_mat)
            lc = lcmodel.lcModel(crowd_res)
            lc.run_em(10**(-10))
            f_lc_save = open(b_dir + img_save_folder + "lc_model.pickle", "w")
            pickle.dump(lc, f_lc_save)
            f_lc_save.close()
            est_labels = lc.estimated_labels(0.5)
            pos_ind_list = np.nonzero(est_labels == 1)[0]
            for i in pos_ind_list:
                patch_info = org_extended[i] # ('./cooking_001.jpg', (0, 0, 25, 25))
                file_id = file_list.index(patch_info[0])
                #if patch_info[1][0] + patch_info[1][2] < img_mask_list[file_id].shape[0] and patch_info[1][1] + patch_info[1][3] < img_mask_list[file_id].shape[1]:
                try:
                    img_mask_list[file_id][patch_info[1][0] : min(patch_info[1][0] + patch_info[1][2], img_mask_list[file_id].shape[0]),
                                           patch_info[1][1] : min(patch_info[1][1] + patch_info[1][3], img_mask_list[file_id].shape[1])] \
                                           += np.ones((min(patch_info[1][2], img_mask_list[file_id].shape[0] - patch_info[1][0]), min(patch_info[1][3], img_mask_list[file_id].shape[1] - patch_info[1][1])))
                except ValueError:
                    sys.stderr.write(str(patch_info))
                    

        else:
            sys.stderr.write("ERROR: quality_control_method must be either LC, MV, or No.")
            exit(0)
        
        # create ground truth images
        sys.stdout.write("\n--- Creating ground truth ---\n")
        pos_ind_list = list(set((org_res_mat_ground_truth > 0).nonzero()[0]))
        for i in pos_ind_list:
            patch_info = org_extended_ground_truth[i]
            file_id = file_list.index(patch_info[0])
            img_mask_list_ground_truth[file_id][patch_info[1][0] : min(patch_info[1][0] + patch_info[1][2], img_mask_list_ground_truth[file_id].shape[0]),
                                   patch_info[1][1] : min(patch_info[1][1] + patch_info[1][3], img_mask_list_ground_truth[file_id].shape[1])] \
                += np.ones((min(patch_info[1][2], img_mask_list_ground_truth[file_id].shape[0] - patch_info[1][0]), \
                            min(patch_info[1][3], img_mask_list_ground_truth[file_id].shape[1] - patch_info[1][1])))
        
        # draw & save images
        sys.stdout.write("\n--- Draw & save images ---\n")
        for i in range(len(file_list)):
            sys.stdout.write("\r" + str(i+1) + "/" + str(len(file_list)) + "-th image")
            sys.stdout.flush()
            img_tmp = np.asarray(img_list[i])
            img_tmp[img_mask_list[i].nonzero()] = (255,0,255)
            cv.SaveImage(b_dir + img_save_folder + str(file_list[i]), cv.fromarray(img_tmp))
        
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        
        ### quality evaluation
        f_list_len = len(file_list)
        tp = np.zeros(f_list_len)
        tn = np.zeros(f_list_len)
        fp = np.zeros(f_list_len)
        fn = np.zeros(f_list_len)
        precision = np.zeros(f_list_len)
        recall = np.zeros(f_list_len)
        fmeasure = np.zeros(f_list_len)
        for i in range(f_list_len):
            tp[i] = np.sum(np.logical_and(img_mask_list[i], img_mask_list_ground_truth[i]))
            tn[i] = np.sum(np.logical_and(np.logical_not(img_mask_list[i]), np.logical_not(img_mask_list_ground_truth[i])))
            fp[i] = np.sum(np.logical_and(img_mask_list[i], np.logical_not(img_mask_list_ground_truth[i])))
            fn[i] = np.sum(np.logical_and(np.logical_not(img_mask_list[i]), img_mask_list_ground_truth[i]))
            if tp[i] + fp[i] != 0:
                precision[i] = float(tp[i]) / float(tp[i] + fp[i])
            else:
                precision[i] = np.nan
            if tp[i] + fn[i] != 0:
                recall[i] = float(tp[i]) / float(tp[i] + fn[i])
            else:
                recall[i] = np.nan
            if precision[i] != np.nan and recall[i] != np.nan and precision[i] != 0 and recall[i] != 0:
                fmeasure[i] = (2.0 * recall[i] * precision[i]) / (precision[i] + recall[i])
            else:
                fmeasure[i] = np.nan
            print str(i) + "-th image: " + "precision = " + str(precision[i]) + ", recall = " + str(recall[i]) + ", F-measure = " + str(fmeasure[i])
            #rect_mat = img_mask_list[i][:, :, np.newaxis] * np.array([[(255,0,255)] * img_list[i].cols] * img_list[i].rows)
            #img_list[i] = cv.fromarray(np.asarray(img_list[i]) + rect_mat)
            #cv.SaveImage(b_dir + argvs[7] + str(file_list[i]), img_list[i])
        
        print "In short,"
        precision_short = float(np.sum(tp)) / float(np.sum(tp) + np.sum(fp))
        recall_short = float(np.sum(tp)) / float(np.sum(tp) + np.sum(fn))
        print "Precision = " + str(precision_short)
        print "Recall = " + str(recall_short)
        print "F-measure = " + str((2.0 * precision_short * recall_short) / (precision_short + recall_short))
        
        if qual_method == "No":
            f_save = open(b_dir + eval_save_folder + "quality_no.pickle", "w")
        elif qual_method == "MV":
            f_save = open(b_dir + eval_save_folder + "quality_mv.pickle", "w")
        elif qual_method == "LC":
            f_save = open(b_dir + eval_save_folder + "quality_lc.pickle", "w")
        else:
            sys.stderr.write("ERROR: quality control method must be either No or MV or LC.")
            exit(0)
        pickle.dump([file_list, precision, recall, fmeasure], f_save)
        f_save.close()
