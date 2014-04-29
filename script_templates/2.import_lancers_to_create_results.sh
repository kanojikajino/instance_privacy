#!/bin/sh

set -e

base_dir=$(cd ../test/100_50_5/ && pwd)"/"
lancers_csv="lancers_task_download_data_test.csv"
num_input="1"
num_answer="1"
quality_control="no"

load_img_dir=$(cd ../test/resized_pics_small/ && pwd)"/"
save_dir=$(cd ../test/ && pwd)"/"
subinstance_size="100"
num_subinstances_to_combine="5"
clickable_size="50"
num_clickable_areas="10" # This must be equal to subinstance_size * num_subinstances_to_combine / clickable_size
variable_name="mosaic"
instance_per_task="10"
prefix_url_for_mosaics=${save_dir}${subinstance_size}"_"${clickable_size}"_"${num_subinstances_to_combine}"/mosaics/"
prefix_url_for_htmls=${save_dir}${subinstance_size}"_"${clickable_size}"_"${num_subinstances_to_combine}"/output/htmls/"

cd ../../../2013/import-lancers
python import-lancers-integrate-output.py ${base_dir}${lancers_csv} ${base_dir} ${num_input} ${num_answer}

cd ../../2014/instance_privacy/postProcessOfInstanceClippingProtocol/
python convert_data.py ${base_dir}"parameters.pickle" ${base_dir}"lancers_result.pickle" ${base_dir}
python draw_results_using_converted_data.py ${base_dir}"parameters.pickle" ${base_dir}"converted_result.pickle" ${quality_control} ${base_dir}
