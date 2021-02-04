work_dir="/data/home/varga/processing_data/new_data_sorted"
prefix_subdirs="sub-"
cp_script="/tmp/tmp.qdPaSxoIoZ/cmake-build-debug-tuplak/bayessian_segmentation_cpp"
label_desc="/data/home/varga/processing_data/label_desk.txt"
train_subjects="/data/home/varga/processing_data/train_subjects.txt"
opts_cnf="/data/home/varga/processing_data/config_ini.txt"
working="/data/home/varga/processing_data/workdir"
modalities_cnf="/data/home/varga/processing_data/modalities.ini"
test_subj="/data/home/varga/processing_data/test_subjects.txt"


export PYTHONPATH="${PYTHONPATH}:/tmp/tmp.qdPaSxoIoZ/cmake-build-debug-tuplak/"
#echo $PYTHONPATH

#echo "$subjects $label_desc $working"
#python 6p_calculate_overlap.py $subjects $label_desc $working
#echo "$subjects $label_desc $opts_cnf $working $modalities_cnf"
echo "$train_subjects $label_desc $opts_cnf $working $modalities_cnf $test_subj"
python 8p_construct_constraints.py $train_subjects $label_desc $opts_cnf $working $modalities_cnf
python 9p_fit.py $train_subjects $label_desc $opts_cnf $working $modalities_cnf $test_subj
python 10p_analyse_result_overlap_loso.py $train_subjects $label_desc  $modalities_cnf $test_subj