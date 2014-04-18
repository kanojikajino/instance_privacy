#!/bin/bash
#
# Create a csv file for Lancers from files.
# ${1} = extension of files (.html, .jpg, etc.)
# ${2} = variable name (that comes in the first row)
# ${3} = path to the load directory.
#    ex) /Users/kjn/Dropbox/Public/20130802
# ${4} = #(instances) per one task
#    ex) 10
# ${5} = prefix
#    ex) https://dl.dropboxusercontent.com/u/17481238/20130705_for_lancers/

if [ $# -ne 5 ]; then
  echo "${0} [extension(.jpg/.html)] [variable_name] [path_to_html_files/] [#(instances)/task] [prefix_for_each_instance]"
  exit 1
fi


cd ${3}
count=0
line=""
for i in `seq 1 ${4}`
do
    line="${line}${2}${i},"
done
echo ${line}$'\b'
line=""
for i in *${1}
do
    line="${line}${5}${i}"
    count=$((count+1))
    if test ${count} -lt ${4}; then
	line="${line},"
    fi
    if test ${count} -eq ${4}; then
	echo $line
	count=0
	line=""
    fi
done
if test ${count} -ne "0"; then
    echo $line
fi
