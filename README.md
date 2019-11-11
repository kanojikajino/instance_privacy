Instance-privacy preserving crowdsourcing
=========================================

This repository contains several programs for the paper "Instance-privacy preserving crowdsourcing", presented in the Second AAAI Conference on Human Computation and Crowdsourcing (HCOMP-14).

## Demo

```bash
cd instance_clipping_protocol
python instance_clipping_and_mixing.py ../_test/input_images/ ../_test 100 50 5 # apply IC protocol to obtain mosaic images composed of subinstances
cd ../submit_crowdsourcing
sh generate_interface_wrapper.sh `cd "../_test/100_50_5/mosaics/"; pwd`/ `cd "../_test/100_50_5/mosaics/"; pwd`/ 50 10 10 `cd "../_test/100_50_5/"; pwd`/ # construct a web interface for annotation
# Run crowdsourcing here
cd ../post_process_for_instance_clipping_protocol
python import_crowd_results.py ../_test/crowdsourcing_result.csv ../_test/ 1 1 # convert results from crowdsourcing into pickle file
python convert_data.py ../_test/100_50_5/parameters.pkl ../_test/workers_result.pickle ../_test/ # convert the pickle file into BinaryData defined in crowd_data.py
python draw_results_using_converted_data.py ../_test/100_50_5/parameters.pkl ../_test/converted_result.pkl mv ../_test/ # draw masked images based on crowdsourced annotations.
```

## Details about scripts
### `instance_clipping_protocol/instance_clipping_and_mixing.py`

This script applies an instance clipping protocol to a set of images to obtain mosaic images, to which crowd workers will give annotations.
It has three parameters: `subinstance_size`, `clickable_size`, and `num_subinstances_to_combine`.

`subinstance_size` determines the size of the clipping window (corresponding to `C` in Figures 1 and 2 in the paper). `clickable_size` determines the size of the target window (corresponding to `A[eta]` in Figure 1, or `S` in Figure 2 in the paper).
`num_subinstances_to_combine` determines the number of subinstances on one side of a combined image. In Figure 2 in the paper, `num_subinstances_to_combine` is 6.


```bash
python instance_clipping_and_mixing.py [path/to/a/folder/containing/jpeg/files] [path/to/output/results] [subinstance_size] [clickable_size] [num_subinstances_to_combine]
```

### `post_process_for_instance_clipping_protocol/import_crowd_results.py`
Convert workers' answers in the csv format into a pickle file.

#### Data format

##### Input csv file

- Each row contains results for one task, where a task is defined as a set of instances.
  - a single task is assigned to a single worker.
  - a single task is composed of multiple instances.
  - a worker will give a single answer to a single instance.
- Each column corresponds to the following items:
  - 1st column corresponds to task ids (not instance ids),
  - 2nd to result ids,
  - 3rd to worker ids,
  - 4th to 4 + `num_input`-th to instance ids
	- instance id should be a path to the instance html file, whose name is `[instance_id].[subinstance_size].html`
  - the rests are results from workers.
    - the format of the result is `[elapsed time] [row id 1]_[col id 1] [row id 2]_[col id 2] ...`

The following is a sample input file:
``` csv
TaskId,ResultId,WorkerName,n1,n2,a1,a2
LTI000495499,LRI003723485,kajino,0.50.html,1.50.html,21.987 04_04 04_06 04_07 05_06 05_07,14.154 00_00 00_02 00_03 01_02 01_03 06_00 07_00 07_01
LTI000495500,LRI003723487,kajino,2.50.html,3.50.html,11.866 00_02 01_02 04_05 04_06 04_07 08_08 09_08,17.642 02_02 02_03 02_05 03_02 03_03 03_05 08_06 08_07 09_06 09_07
LTI000495501,LRI003723492,hiroshi,4.50.html,5.50.html,12.306 02_08 02_09,10.322 04_07 05_07 09_09
LTI000495502,LRI003723494,hiroshi,6.50.html,7.50.html,8.674,19.802 01_02 01_03 02_00 02_01 04_08 06_04 06_05 06_08 07_04 07_05 07_08 08_02
LTI000495503,LRI003723496,kajino,8.50.html,9.50.html,15.529 00_00 00_08 00_09 01_00 06_02 06_03 07_02 07_03 08_00 08_01 09_00 09_01,4.122
LTI000495504,LRI003723498,kajino,10.50.html,11.50.html,8.961 00_03 01_03 02_09 03_09,5.505 06_07
LTI000495505,LRI003723499,hiroshi,12.50.html,13.50.html,12.593 02_04 02_05 03_04 03_05 04_06 04_07 05_06 05_07 08_05,11.513 04_01 05_01 06_00 06_01 09_08
LTI000495506,LRI003723500,hiroshi,14.50.html,0.50.html,2.17,6.61 04_04 04_06 04_07 05_06 05_07
```

##### Output pickle file

1. a list of worker IDs
1. a list of instance IDs
1. an array of results.
   - The result array is a #(instance) x #(workers) array, where each column contains results for each instance.
   - If the worker does not work on an instance, then the corresponding element is None.

In the above sample, the list of worker IDs will be 
```python
worker_ids = ["kajino", "hiroshi"]
```
the list of instance IDs will be
```python
instance_ids = ["0.50.html", "1.50.html", "2.50.html", "3.50.html", "4.50.html", "5.50.html", "6.50.html", "7.50.html", "8.50.html", "9.50.html", "10.50.html", "11.50.html", "12.50.html", "13.50.html", "14.50.html"]
```
and the results array will be
```python
result_array[0, 0] = "21.987 04_04 04_06 04_07 05_06 05_07"
result_array[1, 0] = "14.154 00_00 00_02 00_03 01_02 01_03 06_00 07_00 07_01"
```

### `post_process_for_instance_clipping_protocol/convert_data.py`
This script converts the output of `import_crowd_results.py` into `BinaryData` defined in `crowd_data.py`.

### `post_process_for_instance_clipping_protocol/draw_results_using_converted_data.py`
This script draws masked images based on the outputs of `instance_clipping_and_mixing.py` and `convert_data.py`.

### `post_process_for_instance_clipping_protocol/information_loss.py`
This script comutes information loss from the output of `convert_data.py`


### Other files

#### `lcmodel.py`
This script implements the Dawind & Skene model proposed in 1979.

#### `crowd_data.py`
This script implements a data structure for binary responses from crowd workers.
