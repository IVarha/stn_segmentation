work_dir=$1
bids_folder=$3
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
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
#    echo "$d/$name_1"

    stri=${d#$work_dir//$prefix_subdirs}
    echo "$stri"
    for i in $bids_folder/sub$stri/ses-presurg/anat/*T1w.nii.gz; do
        cp -f $i $d/$name_T1
    done

    for i in $bids_folder/sub$stri/ses-presurg/anat/*T2w.nii.gz; do
        cp -f $i $d/$name_T2
    done
#    for i in $d/sub$stri*.fcsv; do
#        echo $i
#        mv $i $d/fiducials.fcsv
#    done
#


done