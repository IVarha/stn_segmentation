work_dir=$1
prefix_subdirs=$2

pts="czech_points.fcsv"
outfile="surg_ptsT1.fcsv"
lab_file="t2_resampled.nii.gz"

# segment alex data
for d in $work_dir//$prefix_subdirs*;
do
    #echo "$d"
    for tfmf in $d/*.tfm; do
        #echo $tfmf
        python3 ../scripts/test_py_coords.py $d/$pts /home/varga/processing_data/participants-ED2.xlsx.ods $d/$outfile
    done

    #python lps_to_ras.py $d/$pts $d/$seg_out $d/$lab_file
    # WM_mask
    #create mask

done




done
wait