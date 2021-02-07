
import configparser
import bayessian_appearance.settings as settings
import numpy as np
import fsl.transform.flirt as fl
import fsl.data.image as f_im
import os
import json


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