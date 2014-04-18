# -*- coding: utf-8 -*-
""" Data Converter.

usage: python %s <load_img_folder> <loc_list.pickle> <result_mat.pickle> <save_folder/> <threshold>
python convert_data.py test/performance_50/data/org.pickle test/performance_50/data/shuffled.pickle test/performance_50/result_head_task/instance_list.pickle test/performance_50/result_head_task/result_mat.pickle 
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
from ..dataStructure import crowdData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lancers_result.pickle to more friendly data.")
    parser.add_argument("load_img_dir", type=str, help="A directry of original images.")
    parser.add_argument("save_dir", type=str, help="A directory to save clipped results and miscs.")
    parser.add_argument("subinstance_size", type=int, help="The size of a clipping window [pixel].")
    parser.add_argument("num_subinstances_to_combine", type=int, help="The number of subinstances on one side of a combined image.")
    args = parser.parse_args()
    
    command_date = datetime.datetime.now().strftime(u'%Y/%m/%d %H:%M:%S')
    _create_save_dir(args.save_dir, str(args.subinstance_size) + "_" + str(args.num_subinstances_to_combine))
    save_path = os.path.join(args.save_dir, str(args.subinstance_size) + "_" + str(args.num_subinstances_to_combine))
    print args
    print "Command was executed on " + command_date

    
    
    exit(0)
    argvs = sys.argv
    argc = len(argvs)
    if(argc != 7):
        print "usage: python %s <load_img_folder> <org.pickle> <shuffled.pickle> <result_mat.pickle> <instance_list.pickle> <save_folder/>" % argvs[0]
        exit(0)
    else:
        """
        org[0] = ('./cooking_001.jpg', (0, 0, 50, 50))
        shu[0] = ('357.jpg', 5, 8)
        inst[0] = 'https://dl.dropboxusercontent.com/u/17481238/20131003/performance_50/htmls/0.25.html'
        mat[0] = array([ '46.899 06_07 07_07 10_16 10_17 11_16 11_17 12_22 13_22 16_22 16_23 17_22 17_23',
                         None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None, None, None, None, None, None, None,
                         None, None], dtype=object)
        """
        b_dir = os.getcwd() + '/'
        f_org = open(b_dir + argvs[2])
        org = pickle.load(f_org)
        f_org.close()
        f_shu = open(b_dir + argvs[3])
        shu = pickle.load(f_shu)
        f_shu.close()
        
        f_mat = open(b_dir + argvs[4])
        mat = pickle.load(f_mat)
        f_mat.close()
        
        f_inst = open(b_dir + argvs[5])
        inst = pickle.load(f_inst)
        f_inst.close()
        
        # count the number of subinstances in one mosaic
        sqrt_num_sub_in_mos = max([shu[i][1] for i in range(len(shu)) if shu[i] != None and shu[i][0] == "0.jpg"]) + 1
        print "#(subinstances in one side of one mosaic) =", sqrt_num_sub_in_mos
        
        # count the number of mosaics
        num_mos = mat.shape[0]
        num_worker = mat.shape[1]
                
        # arrange each instance in inst & extract the size of a clickalable area & the size of a subinstance area.
        clickalable_area_size = 0
        pat_pir = re.compile('\.')
        for i in range(len(inst)):
            tmp = os.path.basename(inst[i]) # inst[i] = 'https://dl.dropboxusercontent.com/u/17481238/20130930/performance_50/htmls/99.25.html'
            inst[i] = pat_pir.split(tmp)[0] + '.jpg' # '99' + '.jpg'
        clickalable_area_size = int(pat_pir.split(tmp)[1]) # 25
        print "clickalable_area_size =", clickalable_area_size
        subinst_size = org[0][1][2] # org[0] = ('./cooking_001.jpg', (0, 0, 50, 50))
        print "subinst_size =", subinst_size
        division = subinst_size / clickalable_area_size # 2 * 2 clickalable areas in one sub-instance
        
        # create a pos_labels & neg_labels that contain the ids of positive & negative instances
        np.set_printoptions(threshold=np.nan)
        pat = re.compile(' ')
        pat_unsco = re.compile('_')
        pos_labels = []
        neg_labels = []
        all_labels = set([])
        num_pos = 0
        num_neg = 0
        num_all = 0
        for k in range(sqrt_num_sub_in_mos * division):
            for l in range(sqrt_num_sub_in_mos * division):
                all_labels.add((k,l))
        
        for i in range(mat.shape[0]):
            flag = True
            for j in range(mat.shape[1]):
                if mat[i][j] != None and flag:
                    tmp = pat.split(mat[i][j])[1:] # [0] = elapsed time, ["01_00", "02_03"]
                    pos_tmp = set([]) # necessary to calculate neg_labels.
                    for k in tmp:
                        fine_row = int(pat_unsco.split(k)[0])
                        fine_col = int(pat_unsco.split(k)[1])
                        pos_tmp.add((fine_row, fine_col))
                        tup = (inst[i], (fine_row, fine_col), j)
                        pos_labels.append(tup)
                    num_pos += len(pos_tmp)
                    neg_tmp = list(all_labels.difference(pos_tmp))
                    num_neg += len(neg_tmp)
                    num_all += len(all_labels)
                    neg_labels.extend(zip([inst[i]] * len(neg_tmp), neg_tmp, [j] * len(neg_tmp)))
                    #flag = False
                    if len(neg_tmp) + len(pos_tmp) != len(all_labels):
                        sys.stderr.write("ERROR: the number of elements is inconsistent.")
        print "num_pos =", num_pos
        print "num_neg =", num_neg
        print "num_all =", num_all

        # create "new" org list.
        org_extended = set([])
        extention = division**2
        for i in range(len(org)):
            for j in range(division**2):
                org_extended.add((org[i][0], (org[i][1][0] + (j / int(division)) * clickalable_area_size, org[i][1][1] + (j % int(division)) * clickalable_area_size, clickalable_area_size, clickalable_area_size)))
        org_extended = list(org_extended)
        print "len(org_extended) =", len(org_extended)
        
        # load all images
        os.chdir(b_dir + argvs[1])
        file_list = glob.glob('./*.jpg')
        num_files = len(file_list)
        print "#(files) =", num_files
        img_list = [None] * num_files
        img_mask_list = [None] * num_files
        i = 0        
        for file_name in file_list:
            img_list[i] = cv.LoadImageM(file_name)
            img_mask_list[i] = np.zeros((img_list[i].rows, img_list[i].cols))
            file_list[i] = file_name
            i += 1
        os.chdir(b_dir)
        
        # convert shuffled data into original data
        org_res_mat = np.zeros((len(org_extended), mat.shape[1])) # len(org)*(division**2) times #workers matrix
        
        for i in range(len(pos_labels)):
            sys.stdout.write("\r" + str(i+1) + "/" + str(len(pos_labels)))
            sys.stdout.flush()
            try:
                ind = shu.index((pos_labels[i][0], int(pos_labels[i][1][0]) / int(division), int(pos_labels[i][1][1]) / int(division)))
            except ValueError:
                sys.stderr.write("\n" + pos_labels[i][0] + "\n") # the last shuffled image contains black patches that do not appear in the original images.
            org_tmp = org[ind]
            org_res_mat[org_extended.index((org_tmp[0], \
                                            (org_tmp[1][0] + (pos_labels[i][1][0] % division) * clickalable_area_size, \
                                             org_tmp[1][1] + (pos_labels[i][1][1] % division) * clickalable_area_size, \
                                             clickalable_area_size, \
                                             clickalable_area_size))), pos_labels[i][2]] = 1
        sys.stdout.write("\npos finished\n")
        sys.stdout.flush()
        
        for i in range(len(neg_labels)):
            sys.stdout.write("\r" + str(i+1) + "/" + str(len(neg_labels)))
            sys.stdout.flush()
            try:
                ind = shu.index((neg_labels[i][0], int(neg_labels[i][1][0]) / int(division), int(neg_labels[i][1][1]) / int(division)))
            except ValueError:
                sys.stderr.write("\n" + neg_labels[i][0] + "\n") # the last shuffled image contains black patches that do not appear in the original images.
            org_tmp = org[ind]
            org_res_mat[org_extended.index((org_tmp[0], \
                                            (org_tmp[1][0] + (neg_labels[i][1][0] % division) * clickalable_area_size, \
                                             org_tmp[1][1] + (neg_labels[i][1][1] % division) * clickalable_area_size, \
                                             clickalable_area_size, \
                                             clickalable_area_size))), neg_labels[i][2]] = -1
        sys.stdout.write("\nneg finished \n")
        sys.stdout.flush()
        
        f_org_save = open(b_dir + argvs[6] + "org_extended.pickle", "w")
        pickle.dump([org_extended, org_res_mat], f_org_save) # org_extended: relationship between a part of original files and instance id, org_res_mat: labels for each instance id.
        f_org_save.close()
        """
        org_extended[0] = ('./cooking_001.jpg', (0, 0, 25, 25))
        org_res_mat : len(org_extended) * #(workers) numpy.array. each element = {+1, -1, 0}. +1 = pos_label, -1 = neg_label, 0 = not labeled.
        """
        
        exit(0)
