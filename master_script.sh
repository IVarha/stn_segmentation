work_dir="/mnt/f/processing/labels/imaging/bids"
prefix_subdirs="sub"
cp_script="/tmp/tmp.9HaHyiykJ1/cmake-build-debug-remote-host/bayessian_segmentation_cpp"
label_desc="/mnt/f/processing/labels/label_desk.txt"
subjects="/mnt/f/processing/labels/imaging/training_subjects.txt"
opts_cnf="/mnt/f/processing/labels/imaging/config_ini.txt"
working="/mnt/f/processing/labels/imaging/workdir"
modalities_cnf="/mnt/f/processing/labels/imaging/modalities.ini"
test_subj="/mnt/f/processing/labels/imaging/test_subjects.txt"
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


#./1_brain_surface_extract.sh $work_dir $prefix_subdirs
#./2_linear_registration.sh $work_dir $prefix_subdirs
#./3_WM_SEG.sh $work_dir $prefix_subdirs
#./4_intensity_normalisation.sh $work_dir $prefix_subdirs
#./5p_generate_meshes.sh $work_dir $prefix_subdirs $cp_script $label_desc
#python 6p_calculate_overlap.py $subjects $label_desc $working
#python 7p_calculate_norm_intensities.py $subjects $label_desc $opts_cnf $working $modalities_cnf
#python 8p_construct_constraints.py $subjects $label_desc $opts_cnf $working $modalities_cnf
python 9p_fit.py $subjects $label_desc $opts_cnf $working $modalities_cnf $test_subj