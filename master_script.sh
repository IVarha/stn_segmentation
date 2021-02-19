work_dir="/data/home/varga/processing_data/new_data_sorted"
prefix_subdirs="sub-P"
cp_script="/tmp/tmp.qdPaSxoIoZ/cmake-build-debug-tuplak/bayessian_segmentation_cpp"
label_desc="/data/home/varga/processing_data/label_desk.txt"
subjects="/data/home/varga/processing_data/subjects.txt"
opts_cnf="/data/home/varga/processing_data/config_ini.txt"
working="/data/home/varga/processing_data/workdir"
modalities_cnf="/data/home/varga/processing_data/modalities.ini"
test_subj="/data/home/varga/processing_data/test_subjects.txt"
/data/home/varga/processing_data/
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

export PYTHONPATH="${PYTHONPATH}:/tmp/tmp.qdPaSxoIoZ/cmake-build-debug-tuplak/"
echo $PYTHONPATH
#./1_brain_surface_extract.sh $work_dir $prefix_subdirs
#./2_linear_registration.sh $work_dir $prefix_subdirs
#./3_WM_SEG.sh $work_dir $prefix_subdirs
#./4_intensity_normalisation.sh $work_dir $prefix_subdirs
./5p_generate_meshes.sh $work_dir $prefix_subdirs $cp_script $label_desc
#echo "$subjects $label_desc $working"
#python 6p_calculate_overlap.py $subjects $label_desc $working
#echo "$subjects $label_desc $opts_cnf $working $modalities_cnf"
python 7p_calculate_norm_intensities.py $subjects $label_desc $opts_cnf $working $modalities_cnf
#echo "$subjects $label_desc $opts_cnf $working $modalities_cnf $test_subj"
#python 8p_construct_constraints.py $subjects $label_desc $opts_cnf $working $modalities_cnf
#python 9p_fit.py $subjects $label_desc $opts_cnf $working $modalities_cnf $test_subj