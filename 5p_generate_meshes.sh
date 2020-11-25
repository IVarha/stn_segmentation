work_dir=$1

prefix_subdirs=$2

cp_program=$3
labels_desc=$4
#ssss
#while  IFS= read -r line
#do
#    echo "$line"
#    name=$line
#    break
#done < "$im_name"
labels_file="labels_clean.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    $cp_program -run mesh_shrinkage -i $d/$labels_file -labeldesk $labels_desc -workdir $d
    # WM_mask
    #create mask

done