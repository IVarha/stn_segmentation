# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys

import nibabel as nib
import nrrd
import numpy as np
import bayessian_appearance.mesh as mesh
import bayessian_appearance.utils as util
import os
import sys
import vtk
import fsl.transform.flirt as fl
import fsl.data.image as fim
import bayessian_appearance.vt_image as vtim
import bayessian_appearance.nifti_mask as nm
import pandas
import csv


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def concatenate_intensities(x1,x2):
    res = [ ]
    if x1 == []: return x2
    for i in range(len(x2)):
        res.append( x1[i] + x2[i])
    return res

def apply_transf_2_norms( norms,transf):
    a = np.array(norms)


    res = []
    for i in range(a.shape[0]):
        tmp = []
        for j in range(a.shape[1]):
            pt_norm = np.array(list(a[i,j,:]) + [1])
            pt_tmp = list((np.dot(transf,pt_norm))[:3])
            tmp.append(pt_tmp)
        res.append(tmp)
    return res

def calc_intensities( norms,image):
    a = np.array(norms)
    res = []
    for i in range(a.shape[0]):
        tmp = []
        for j in range(a.shape[1]):
            pt_norm = list(a[i,j,:])
            intens  = image.interpolate(pt_norm)
            tmp.append(intens)
        res.append(tmp)
    return res


def norms_2_coords(normals):
    res = []

    for i in range(len(normals)):
        tmp = []
        for j in range(len(normals[0])):
            tmp = tmp + normals[i][j]
        res.append(tmp)
    return res

def calculate_mask_touches( mask, norms):
    res = []
    for i in range(len(norms)):
        tmp = []
        for j in range(len(norms[0])):
            tmp.append( mask.check_neighbours_world(voxel=norms[i][j]))
        res.append(tmp)
    return res


def save_intensities_csv(pdm, filename):
    try:
        os.remove(filename)
    except:
        pass
    f = open(filename,'w')
    wr = csv.writer(f)
    wr.writerows(pdm)
    f.close()

def calculate_intensites_subject(modalities,labels,subject, discretisation, norm_len):

    images= []
    for i in range(len( modalities)):
        im = vtim.Image(subject + os.sep + modalities[i][1])
        im.setup_bspline(3)
        images.append( [modalities[i][0], im])

    im_ref = fim.Image(subject + os.sep + "t1_acpc_extracted.nii.gz")
    im_mni = fim.Image(subject + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")

    forward_transf_fsl = fl.readFlirt(subject + os.sep + "combined_affine_t1.mat")

    to_mni = fl.fromFlirt(forward_transf_fsl, src=im_ref, ref=im_mni, from_="world", to="world")
    from_mni = np.linalg.inv(to_mni)

    lab_im = fim.Image(subject + os.sep + "labels_clean.nii.gz")
    tr_World_voxel = lab_im.getAffine("world", "voxel")

    mask_im = nm.ni_mask(subject + os.sep + "t2_mask.nii.gz")
    for i in range(len(labels)):
        surf = mesh.Mesh(filename=subject + os.sep + labels[i] + "_1.obj")  # read mesh
        surf.apply_transform(mat=to_mni)

        mni_norms = surf.generate_normals(norm_len, discretisation)
        norms_native = apply_transf_2_norms(mni_norms, from_mni)
        #calculate overlap of normal with mask in world coords
        mask_norm = calculate_mask_touches(mask=mask_im,norms= norms_native)
        norms_native = apply_transf_2_norms(norms_native, tr_World_voxel)

        #calculate intensities mask,1st_modal,2nd...
        profiles = mask_norm
        for j in range(len(images)):

            profile= calc_intensities(norms_native, images[j][1])
            profiles = concatenate_intensities(profiles,profile)
        norm_vecs = norms_2_coords(normals=mni_norms)

        # calc result mat
        res = concatenate_intensities(norm_vecs,profiles)
        save_intensities_csv(pdm=res,filename=subject+os.sep+labels[i]+"_profiles.csv")







def parse_conf(conf_file_name):
    conf = util.read_config_ini(conf_file_name)


    return conf




def main_proc(train, label_names, config_name,modalities, workdir):
    tr_subjects = util.read_train_subjects(train)
    labels = util.read_label_desc(label_names)

    mod = util.read_modalities_config(modalities_name)
    cnf = parse_conf(config_name)

    meshes = []
    for sub_i in range(len(tr_subjects)):
        # read im

        calculate_intensites_subject(modalities=mod,labels=labels,subject=tr_subjects[sub_i],
                                     discretisation=cnf['discretisation'],norm_len=cnf['norm_length'])





    meshes = np.array(meshes)
    m_mx = []



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    train_subjects_file = sys.argv[1]
    labels_desc_file = sys.argv[2]
    conf_file = sys.argv[3]
    outp = sys.argv[4]
    modalities_name = sys.argv[5]
    mod = util.read_modalities_config(modalities_name)
    main_proc(train_subjects_file, labels_desc_file, conf_file,modalities=modalities_name, workdir=outp)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
