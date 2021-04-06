work_dir=$1
prefix_subdirs=$2

seg="relabelled.nii.gz"
seg_out="relabelled_clean.nii.gz"
lab_file="t2_resampled.nii.gz"

# segment alex data
for d in $work_dir//$prefix_subdirs*;
do
    echo "$d"
    python lps_to_ras.py $d/$seg $d/$seg_out $d/$lab_file &
    # WM_mask
    #create mask

done




done
wait