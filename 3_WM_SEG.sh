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
t2_file="t2_acpc_wt1.nii.gz"
name_1="t1_acpc_extracted.nii.gz"
out_wm_mask="t2_WM_mask.nii.gz"
out_wm_mask2="t2_mask.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    #echo "$d/$name_1"
    #do FAST
    fast -R 0.0 -H 0.0 $d/$name_1
    # WM_mask
    python wm_mask_slab.py $d/$t2_file $d/$pveseg $d/$out_wm_mask $d/$out_wm_mask2
    #create mask

done