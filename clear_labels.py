# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from bids import BIDSLayout
from bids.tests import get_test_data_path
import os
import sys
import glob
import nrrd
import nibabel as nib
import numpy as np
import skimage.morphology as morph

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def get_coord_voxel(coord_orig,orig_transf):
    # get v
    vox = np.array(coord_orig + [1])

    res = np.dot(orig_transf,vox)
    return res

def clear_labels(t2_img):
    res_mask = np.zeros(t2_img.shape)

    tmp = t2_img.astype(np.int)
    un = np.unique(t2_img.astype(np.int))
    res = np.zeros(tmp.shape)


    for i in un:
        if i != 0:
            msk1 = tmp == i
            msk = morph.remove_small_objects(ar=msk1,min_size=20)
            res[msk] = i
            if ~((msk == msk1).min()):
                print("reached some deletion of small objects")
    return res


    res


    mn = t2_img[res_mask>0].mean()*0.1
    res_mask[t2_img<mn]=0

    return res_mask

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    labels_in = sys.argv[1]
    outp = sys.argv[2]


    #read pve segmentation
    labels_file = nib.load(labels_in)


    pve_transf = labels_file.affine

    res = clear_labels(t2_img=labels_file.get_fdata())

    nif2 = nib.Nifti1Image(res.astype(np.int), pve_transf)
    nib.save(nif2,outp)
    # Initialize the layout
    print(1222)

    # Print some basic information about the layout


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
