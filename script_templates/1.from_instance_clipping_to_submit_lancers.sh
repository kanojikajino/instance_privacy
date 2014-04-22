#!/bin/sh

set -e

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

cd ../instanceClippingProtocol
python instance_clipping_and_mixing.py $load_img_dir $save_dir $subinstance_size $clickable_size $num_subinstances_to_combine
cd ../submitCrowdsourcing/
sh generate_interface_wrapper.sh ${save_dir}${subinstance_size}"_"${clickable_size}"_"${num_subinstances_to_combine}"/mosaics/" $prefix_url_for_mosaics $clickable_size $num_clickable_areas $num_clickable_areas ${save_dir}${subinstance_size}"_"${clickable_size}"_"${num_subinstances_to_combine}"/"
sh make_csv.sh ".html" $variable_name ${save_dir}${subinstance_size}"_"${clickable_size}"_"${num_subinstances_to_combine}"/output/htmls/" $instance_per_task $prefix_url_for_htmls > ${save_dir}${subinstance_size}"_"${clickable_size}"_"${num_subinstances_to_combine}"/lancers_input.csv"
