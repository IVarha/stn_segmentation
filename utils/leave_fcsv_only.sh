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
name_T1="t1_precl_RAW.nii.gz"
name_T2="t2_precl_RAW.nii.gz"
name_1="t1.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    echo "$d/$name_1"

    stri=${d#$work_dir//$prefix_subdirs}
    echo "$stri"
#    for i in $d/sub$stri*.gz; do
#        rm -f $i
#    done
    for i in $d/sub$stri*.fcsv; do
        echo $i
        mv $i $d/fiducials.fcsv
    done
    stri=${d#$work_dir//$prefix_subdirs}
    echo "$stri"
    for i in $d/*T1w_DN.nii.gz; do
        mv -f $i $d/$name_T1
    done

    for i in $d/*T2w_DN.nii.gz; do
        mv -f $i $d/$name_T2
    done


done