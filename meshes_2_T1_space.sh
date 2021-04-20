work_dir=$1
refer=$3
labels=$4
prefix_subdirs=$2
im_name=$work_dir/names
#ssss
#while  IFS= read -r line
#do

export PYTHONPATH="${PYTHONPATH}:/tmp/tmp.qdPaSxoIoZ/cmake-build-debug-tuplak/"
#    echo "$line"
#    name=$line
#    break
#done < "$im_name"
name="mri.nii.gz"
name_1="t1.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    #echo "$d/$name_1"
    #
#    stri=${d#$work_dir//$prefix_subdirs}
#    echo "$stri"
#    for i in $d/sub$stri*T1w.nii.gz; do
#        echo $i
#        cp $i $d/t1_precl_RAW.nii.gz
#    done
#    for i in $d/rsub$stri*T2w.nii.gz; do
#        echo $i
#        cp $i $d/t2_precl_wt1.nii.gz
#    done
#
#    for i in $d/sub$stri*T2w.nii.gz; do
#        echo $i
#        cp $i $d/t2_precl_RAW.nii.gz
#    done
    echo $d

    python meshes_2_t1_space.py $d/1_1.obj $d/transformACPC $d/1_1T1.obj
    python meshes_2_t1_space.py $d/2_1.obj $d/transformACPC $d/2_1T1.obj
    python meshes_2_t1_space.py $d/3_1.obj $d/transformACPC $d/3_1T1.obj
    python meshes_2_t1_space.py $d/4_1.obj $d/transformACPC $d/4_1T1.obj
    python meshes_2_t1_space.py $d/5_1.obj $d/transformACPC $d/5_1T1.obj
    python meshes_2_t1_space.py $d/6_1.obj $d/transformACPC $d/6_1T1.obj
    python meshes_2_t1_space.py $d/1_fitted.obj $d/transformACPC $d/1_fittedT1.obj
    python meshes_2_t1_space.py $d/2_fitted.obj $d/transformACPC $d/2_fittedT1.obj
    python meshes_2_t1_space.py $d/3_fitted.obj $d/transformACPC $d/3_fittedT1.obj
    python meshes_2_t1_space.py $d/4_fitted.obj $d/transformACPC $d/4_fittedT1.obj
    python meshes_2_t1_space.py $d/5_fitted.obj $d/transformACPC $d/5_fittedT1.obj
    python meshes_2_t1_space.py $d/6_fitted.obj $d/transformACPC $d/6_fittedT1.obj



#    echo "$d/t1_precl_RAW.nii.gz $d/fiducials.fcsv $d/t1_acpc.nii.gz $d/transformACPC"
#    python new_data_to_ac_pc.py $d/t1_precl_RAW.nii.gz $d/fiducials.fcsv $d/t1_acpc.nii.gz $d/transformACPC
#    python new_data_to_ac_pc.py $d/t2_precl_RAW.nii.gz $d/fiducials.fcsv $d/t2_acpc_wt1.nii.gz $d/transformACPC
##    mri_gcut $d/$name_1 $d/t1_extracted.nii.gz
done