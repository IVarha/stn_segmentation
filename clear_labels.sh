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
seg="labels.nii.gz"
seg_out="labels_clean.nii.gz"
lab_file="t2_resampled.nii.gz"

# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    python clear_labels.py $d/$seg $d/$seg_out $d/$lab_file
    # WM_mask
    #create mask

done