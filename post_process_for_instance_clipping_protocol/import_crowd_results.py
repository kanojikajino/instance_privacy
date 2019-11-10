#!/Library/Frameworks/EPD64.framework/Versions/Current/bin/python
# -*- coding: utf-8 -*-
""" import crowdsourced data.
The input csv file is assumed to be in the following format.

- each row contains results for one task, where a task is defined as a set of instances.
  - a single task is assigned to a single worker.
  - a single task is composed of multiple instances.
  - a worker will give a single answer to a single instance.
- each column corresponds to the following items:
  - 1st column corresponds to task ids (not instance ids),
  - 2nd  to result ids,
  - 3rd to worker ids,
  - 4th to 4 + `num_input`-th to instance ids
  - the rests are results from workers.

Output pickle file is composed of three objects.
The first one is a list of worker IDs, the second one is a list of instance IDs, and the third one is an array of results.
The result array is a #(instance) x #(workers) array, where each column contains results for each instance.
If the worker does not work on an instance, then the corresponding element is None.

usage: python %s <result.csv> <output_folder/>
REMARK:
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2013/08/03"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2013 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import numpy as np
import pickle
import sys
import os
import argparse
import datetime

def main():
    parser = argparse.ArgumentParser(description="import a csv file obtained from a crowdsourcing platform into a pickle file.")
    parser.add_argument("csv_file", type=str, help="A csv file obtained from the platform.")
    parser.add_argument("save_dir", type=str, help="A directory to save a pickle file.")
    parser.add_argument("num_input", type=int, help="The number of inputs per one instance.")
    parser.add_argument("num_answer", type=int, help="The number of answers per one instance.")
    args = parser.parse_args()

    command_date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(args)
    print("Command was executed on " + command_date)

    task_array = np.loadtxt(args.csv_file, delimiter=',', dtype=str)

    # give IDs for workers
    worker_ids = list(set(task_array[1:, 2])) # the zero-th row corresponds to a header.
    print("#(workers) =", len(worker_ids))

    # give IDs for instances
    num_instances_in_line = (task_array.shape[1] - 3) / (args.num_input + args.num_answer)
    print('#(instances/task) =', num_instances_in_line)
    instance_ids = []

    if args.num_input != 1:
        for i in range(task_array.shape[0] - 1):
            for ii in range(num_instances_in_line):
                if not [task_array[i + 1,  3 + ii * args.num_input + iii] for iii in range(args.num_input)] in instance_ids:
                    instance_ids.append([task_array[i + 1,  3 + ii * args.num_input + iii] for iii in range(args.num_input)])
    else:
        for i in range(task_array.shape[0] - 1):
            for ii in range(num_instances_in_line):
                if not task_array[i + 1,  3 + ii * args.num_input] in instance_ids:
                    instance_ids.append(task_array[i + 1,  3 + ii * args.num_input])

    print('#(instances) =', len(instance_ids))

    # import into numpy.array
    result_array = np.array([[None] * len(worker_ids)] * len(instance_ids))

    if args.num_answer != 1:
        for i in range(task_array.shape[0] - 1):
            worker_id = worker_ids.index(task_array[i + 1, 2])
            for ii in range(num_instances_in_line):
                if args.num_input != 1:
                    instance_id = instance_ids.index([task_array[i + 1,  3 + ii * args.num_input + iii] for iii in range(args.num_input)])
                else:
                    instance_id = instance_ids.index(task_array[i + 1,  3 + ii * args.num_input])
                result_array[instance_id][worker_id] = [task_array[i + 1,
                                                                   3 + num_instances_in_line * args.num_input + ii * args.num_answer + iii]\
                                                        for iii in range(args.num_answer)]
    else:
        for i in range(task_array.shape[0] - 1):
            worker_id = worker_ids.index(task_array[i + 1, 2])
            for ii in range(num_instances_in_line):
                if args.num_input != 1:
                    instance_id = instance_ids.index([task_array[i + 1,  3 + ii * args.num_input + iii] for iii in range(args.num_input)])
                else:
                    instance_id = instance_ids.index(task_array[i + 1,  3 + ii * args.num_input])
                result_array[instance_id][worker_id] = task_array[i + 1,
                                                                  3 + num_instances_in_line * args.num_input + ii * args.num_answer]
    f = open(os.path.join(args.save_dir, 'workers_result.pickle'), 'wb')
    pickle.dump((worker_ids, instance_ids, result_array), f)
    f.close()

if __name__ == "__main__":
    main()
