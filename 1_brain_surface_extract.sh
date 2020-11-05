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
name_1="t1_acpc.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    #echo "$d/$name_1"
    #

    ./bfc_h.sh $d/$name_1 $d/t1_bfc_ac_pc.nii.gz
    rm -rf $d/temp_t1_acpc_extracted.nii.gz
    rm -rf $d/t1_acpc_extracted.nii.gz
    bet $d/t1_bfc_ac_pc.nii.gz $d/t1_acpc_extracted.nii.gz -R -S -B
    #-B
#    runROBEX.sh $d/t1_bfc_ac_pc.nii.gz $d/t1_acpc_extracted.nii.gz
#    mri_gcut $d/$name_1 $d/t1_acpc_extracted.nii.gz -T 0.95
done