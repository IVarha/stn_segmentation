work_dir=$1
refer=$3
labels=$4
prefix_subdirs=$2
im_name=$work_dir/names
#ssss
#while  IFS= read -r line
#do
#    echo "$line"
#    name=$line
#    break
#done < "$im_name"
name="mri.nii.gz"
name_1="t1_acpc_extracted.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    #echo "$d/$name_1"
    #
    fast -R 0.0 -H 0.0 $d/$name_1


done