
import configparser
import bayessian_appearance.settings as settings
import numpy as np
import fsl.data.image as fim
import fsl.transform.flirt as fl
import fsl.data.image as f_im
import os
import json
import vtk
import csv
import bayessian_appearance.vt_image as vtim
import bayessian_appearance.nifti_mask as nm
import ExtPy



def read_label_desc(file_name):
    fm = open(file=file_name,mode='rt')
    res = []
    for sub in fm:
        a = sub.split(',')
        res.append(a[0])
    settings.settings.all_labels = res
    fm.close()
    return res

def read_subjects(file_name):
    fm = open(file=file_name,mode='rt')
    res = []
    for sub in fm:
        res.append(str(sub.strip('\n')))
    fm.close()
    return res




def read_config_ini(file_name):
    fm = open(file=file_name,mode='rt')
    res = {}
    for sub in fm:
        line= str(sub.strip('\n'))
        line=line.split(',')
        res.update({line[0]:line[1]})
    fm.close()

    res['norm_length'] = float(res['norm_length'])
    settings.settings.norm_length = res['norm_length']
    res['discretisation'] = int(res['discretisation'])
    settings.settings.discretisation = res['discretisation']
    return res

def read_modalities_config(file_name):
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    res = [ ]
    keys = [x for x in cfg['modalities']]
    for i in range(len(keys)):
        res.append( [keys[i],cfg['modalities'][keys[i]]])
    settings.settings.modalities = res
    return res

def read_segmentation_config(file_name):
    cfg = configparser.ConfigParser()
    cfg.read(file_name)

    res = {}
    keys = [x for x in cfg['segmentation_conf']]
    for i in range(len(keys)):
        res.update({keys[i]: cfg['segmentation_conf'][keys[i]]})

    if 'use_constraint' in res.keys():
        if res['use_constraint'] == 'True':
            res['use_constraint']= True

        else:
            res['use_constraint'] = False
            #res.append( [keys[i],cfg['segmentation_conf'][keys[i]]])
        settings.settings.use_constraint = res['use_constraint']



    settings.settings.atlas_dir = res['atlas_dir']
    res['labels_to_segment'] = res['labels_to_segment'].split(',')
    settings.settings.labels_to_segment = res['labels_to_segment']
    if 'dependent_constraint' in res.keys():
        settings.settings.dependent_constraint = json.loads(res['dependent_constraint'])
    else:
        settings.settings.dependent_constraint = []
    return res

def apply_transf_2_pts(pts,transf):
    res = []
    for i in range(len(pts)):

        pt_norm = np.array(pts[i] + [1])
        pt_tmp = list((np.dot(transf,pt_norm))[:3])

        res.append(pt_tmp)
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


def read_fsl_mni2native_w(subject):
    im_mni = f_im.Image(subject + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")
    im_native = f_im.Image(subject + os.sep + "t1_acpc_extracted.nii.gz")

    fsl_mni2nat = fl.readFlirt(subject + os.sep + "combined_affine_reverse.mat")

    mni_w = fl.fromFlirt(fsl_mni2nat,im_mni,im_native,'world','world')
    return mni_w

def read_fsl_native2mni_w(subject):
    a = read_fsl_mni2native_w(subject)
    return np.linalg.inv(a)

# method for compute position of label in data based on label
def comp_posit_in_data(label):
    return settings.settings.all_labels.index(label)

#generates point array
def generate_mask(x_r,y_r,z_r,dt):

    x_min = x_r[0]
    x_max = x_r[1]

    x_arr = []
    i = 0
    while True:
        x_arr.append(x_min + i * dt)
        if (x_min + i * dt) >= x_max:
            break
        i += 1
    y_min = y_r[0]
    y_max = y_r[1]
    y_arr = []
    i = 0
    while True:
        y_arr.append(y_min + i * dt)
        if (y_min + i * dt) >= y_max:
            break
        i += 1

    z_min = z_r[0]
    z_max = z_r[1]
    z_arr = []
    i = 0
    while True:
        z_arr.append(z_min + i * dt)
        if (z_min + i * dt) >= z_max:
            break
        i += 1

    res_arr = np.zeros((len(x_arr),len(y_arr),len(z_arr),3))
    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            for k in  range(len(z_arr)):
                res_arr[i,j,k,0] = x_arr[i]
                res_arr[i, j, k, 1] = y_arr[j]
                res_arr[i, j, k, 2] = z_arr[k]
    return res_arr

#########7p
def concatenate_intensities(x1,x2):
    res = [ ]
    if x1 == []: return x2
    for i in range(len(x2)):
        res.append( x1[i] + x2[i])
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

def calculate_intensites_subject(modalities,labels,subject, discretisation, norm_len,mesh_name_end):

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
        surf = ExtPy.cMesh(subject + os.sep + labels[i] + mesh_name_end)  # read mesh
        surf.apply_transform(to_mni.tolist())

        volum = surf.calculate_volume()

        mni_norms = surf.generate_normals(norm_len, discretisation)
        norms_native = apply_transf_2_norms(mni_norms, from_mni)
        #calculate overlap of normal with mask in world coords
        mask_norm = calculate_mask_touches(mask=mask_im,norms= norms_native)
        norms_native = apply_transf_2_norms(norms_native, tr_World_voxel)
        surf.apply_transform(from_mni.tolist())
        #calc points
        mp = surf.generate_mesh_points(20)


        #calculate intensities mask,1st_modal,2nd...
        profiles = mask_norm
        means = []
        for j in range(len(images)):

            profile= calc_intensities(norms_native, images[j][1])
            mp2 = apply_transf_2_pts(mp,images[j][1]._world_2_vox)
            ######intensity blok(-mean)
            mn = np.array(images[j][1].interpolate_list(mp2)).mean()
            means.append(mn)
            #mn = 0 # for clean mean
            profiles = concatenate_intensities(profiles,(np.array(profile) - mn).tolist())
        norm_vecs = norms_2_coords(normals=mni_norms)
        means = means + [volum]
        # calc result mat
        res = concatenate_intensities(norm_vecs,profiles)
        save_intensities_csv(pdm=res + [means],filename=subject+os.sep+labels[i]+"_profiles.csv")


