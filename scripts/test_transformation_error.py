import sys
import csv
import nibabel as nib
import numpy as np
import pickle
import h5py
import fsl.transform.flirt as flirt
import fsl.data.image as fsl_data
import scipy.io as sc_io



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inp_mni_image = sys.argv[1]

    input_native_image = sys.argv[2]
    transformation = sys.argv[3]
    im_t2 = sys.argv[4]
    outp_file = sys.argv[5]
    #remove
    a = nib.load(input_native_image)
    a.get_qform()
    transfor = flirt.readFlirt(transformation)
    in_ni = fsl_data.Image(input_native_image)
    in_mni = fsl_data.Image(inp_mni_image)

    res_trans = flirt.fromFlirt(transfor,in_mni,in_ni,'world','world')


    in_mni_res = nib.load(inp_mni_image)

    t2_f = nib.load(im_t2)
    mni_aff = in_mni_res.affine
    nif2 = nib.Nifti1Image(t2_f.get_fdata(), np.dot(res_trans,t2_f.affine))
    nib.save(nif2, outp_file)
    # Initialize the layout
    print(1222)

    # Print some basic information about the layout
