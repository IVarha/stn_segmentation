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
echo "LINREG #######################################"


for d in $work_dir/$prefix_subdirs*; do
    echo "$d"
    #linear reg
    flirt -in $d/$name_1 -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out $d/t1_brain_to_mni_affine -omat $d/affine_t1.mat -dof 12 &


done

wait

for d in $work_dir//$prefix_subdirs*; do
    #2nd stage
    flirt -in $d/t1_brain_to_mni_affine.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out $d/t1_brain_to_mni_stage2 -omat $d/affine_t1_stage2.mat -nosearch -refweight $FSLDIR/data/standard/MNI152lin_T1_1mm_subbr_mask.nii.gz &

done

wait

for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    #echo "$d/$name_1"
    #

    convert_xfm -omat $d/combined_affine_t1.mat -concat $d/affine_t1_stage2.mat $d/affine_t1.mat
    convert_xfm -omat $d/combined_affine_reverse.mat -inverse $d/combined_affine_t1.mat
#    mri_gcut $d/$name_1 $d/t1_acpc_extracted.nii.gz -T 0.95
done


for d in $work_dir//$prefix_subdirs*; do

    flirt -in $d/$name_1 -out $d/t1_brain_to_mni_stage2_apply -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -applyxfm -init $d/combined_affine_t1.mat &
#    mri_gcut $d/$name_1 $d/t1_acpc_extracted.nii.gz -T 0.95
done
wait
#for d in $work_dir//$prefix_subdirs*; do
#    echo "$d"
#    #echo "$d/$name_1"
#    #
#
#    flirt -in $d/$name_1 -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out $d/t1_brain_to_mni_affine -omat $d/affine_t1.mat -dof 12
#
#
#    flirt -in $d/t1_brain_to_mni_affine.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out $d/t1_brain_to_mni_stage2 -omat $d/affine_t1_stage2.mat -nosearch -refweight $FSLDIR/data/standard/MNI152lin_T1_1mm_subbr_mask.nii.gz
#
#    convert_xfm -omat $d/combined_affine_t1.mat -concat $d/affine_t1_stage2.mat $d/affine_t1.mat
#    convert_xfm -omat $d/combined_affine_reverse.mat -inverse $d/combined_affine_t1.mat
#
#    flirt -in $d/$name_1 -out $d/t1_brain_to_mni_stage2_apply -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -applyxfm -init $d/combined_affine_t1.mat
#
##    mri_gcut $d/$name_1 $d/t1_acpc_extracted.nii.gz -T 0.95
#done
