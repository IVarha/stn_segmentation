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
import skimage.morphology as moph
import scipy.ndimage.morphology as morph2

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def get_coord_voxel(coord_orig,orig_transf):
    # get v
    vox = np.array(coord_orig + [1])

    res = np.dot(orig_transf,vox)
    return res

def create_mask(t2_img,t2_transf,pve_img, pve_transf):
    res_mask = np.zeros(t2_img.shape) #ma
    res_mask_full = np.zeros(t2_img.shape) # mask of t2 image
    brain_mask = np.zeros(t2_img.shape)
    revers_pve = np.linalg.inv(pve_transf)

    for i in range(t2_img.shape[0]):
        for j in range(t2_img.shape[1]):
            for k in range(t2_img.shape[2]):
                if t2_img[i,j,k] > 0.01:
                    vox_ras = get_coord_voxel([i,j,k],t2_transf)
                    pos_xyz = np.dot(revers_pve,vox_ras)
                    pos = np.round(pos_xyz).astype(np.int)
                    t_res = 0
                    br_vox = 0
                    try:
                        if pve_img[pos[0],pos[1],pos[2]] == 3:
                            t_res  = 1
                        if pve_img[pos[0],pos[1],pos[2]]>0:
                            br_vox = 1
                    except:
                        pass
                    res_mask_full[i,j,k] = 1
                    res_mask[i,j,k] = t_res
                    brain_mask[i,j,k] = br_vox
    mn = t2_img[res_mask>0].mean()*0.1
    res_mask[t2_img<mn]=0
    res_mask= moph.remove_small_holes(res_mask.astype(np.int),area_threshold=20)
    res_mask = morph2.binary_fill_holes(res_mask)
    res_mask = morph2.binary_dilation(res_mask,iterations=1)
    res_mask = morph2.binary_erosion(res_mask, iterations=1)
    return [res_mask.astype(np.int),res_mask_full.astype(np.int),res_mask_full.astype(np.int)]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    t2_in = sys.argv[1]
    pve_seg = sys.argv[2]
    outp = sys.argv[3]

    outp_fullmask = sys.argv[4]
    outp_brain_mask = sys.argv[5]
    #read pve segmentation
    pve_file = nib.load(pve_seg)

    t2_file = nib.load(t2_in)

    pve_transf = pve_file.affine
    t2_transf = t2_file.affine

    res = create_mask(t2_img=t2_file.get_fdata(),t2_transf=t2_transf,
                      pve_img=pve_file.get_fdata(),pve_transf=pve_transf)

    nif2 = nib.Nifti1Image(res[0], t2_transf)
    nib.save(nif2,outp)
    nif2 = nib.Nifti1Image(res[1], t2_transf)
    nib.save(nif2,outp_fullmask)
    nif2 = nib.Nifti1Image(res[2], t2_transf)
    nib.save(nif2, outp_brain_mask)
    # Initialize the layout
    print(1222)

    # Print some basic information about the layout


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
