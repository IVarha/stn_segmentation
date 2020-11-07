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
pveseg="t1_acpc_extracted_pveseg.nii.gz"
t2_file="t2_resampled.nii.gz"
out_t2="t2_acpc_normalised.nii.gz"
wm_mask="t2_WM_mask.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    fcm-normalize -i $d/$t2_file -tm $d/$wm_mask -o $d/$out_t2 -v -c t2 -s -p
    # WM_mask
    #create mask

done