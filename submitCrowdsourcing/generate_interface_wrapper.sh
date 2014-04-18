#!/bin/sh

set -e

if [ $# -ne 6 ]; then
  echo "${0} [input_image_FULL_path/] [prefix_url/] [clickable_size] [num_clickable_areas_on_col] [num_clickable_areas_on_row] [save_FULL_path/]"
  exit 1
fi

cd generate_interface_from_single_image

for i in `ls ${1}*.jpg`
do
    python generate_html.py ${2}`basename ${i}` ${3} ${4} ${5} 2
done

cp -r output ${6}
cd output/htmls
for i in `ls *.html`
do
    rm ${i}
done
