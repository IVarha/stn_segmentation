work_dir=$1
prefix_subdirs=$2

seg="labels.nii.gz"
seg_out="labels_clean.nii.gz"
lab_file="t2_resampled.nii.gz"

# segment alex data
for d in $work_dir//$prefix_subdirs*;
do
    echo "$d"
    python clear_labels.py $d/$seg $d/$seg_out $d/$lab_file &
    # WM_mask
    #create mask

done


for d in $work_dir//$prefix_subdirs*; do
    echo "$d" >> $work_dir/subjects.txt
    #python clear_labels.py $d/$seg $d/$seg_out $d/$lab_file
    # WM_mask
    #create mask

done