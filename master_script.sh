work_dir=$1
refer=$3
labels=$4
prefix_subdirs=$2
cp_script="/tmp/tmp.9HaHyiykJ1/cmake-build-debug-remote-host/bayessian_segmentation_cpp"
label_desc="/mnt/f/processing/labels/label_desk.txt"
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


./1_brain_surface_extract.sh $work_dir $prefix_subdirs
./2_linear_registration.sh $work_dir $prefix_subdirs
./3_WM_SEG.sh $work_dir $prefix_subdirs
./4_intensity_normalisation.sh $work_dir $prefix_subdirs
./5p_generate_meshes.sh $work_dir $prefix_subdirs $cp_script $label_desc