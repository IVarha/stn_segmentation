import configparser
import csv
import json
import math
import os


import fsl.data.image as f_im
import fsl.data.image as fim
import fsl.transform.flirt as fl
import numpy as np

import bayessian_appearance.nifti_mask as nm
import bayessian_appearance.settings as settings
import bayessian_appearance.vt_image as vtim
import ExtPy

def read_label_desc(file_name):
    fm = open(file=file_name, mode='rt')
    res = []

    prec = []
    for sub in fm:
        a = sub.split(',')
        res.append(a[0])
        if len(a)>2:
            if a[2] != '':
                prec.append(a[2])
            else:
                prec.append("default")
        else:
            prec.append("default")
    settings.settings.pca_precision_labels = prec
    settings.settings.all_labels = res
    fm.close()
    return res


def read_subjects(file_name):
    fm = open(file=file_name, mode='rt')
    res = []
    for sub in fm:
        res.append(str(sub.strip('\n')))
    fm.close()
    return res


def read_config_ini(file_name):
    fm = open(file=file_name, mode='rt')
    res = {}
    for sub in fm:
        line = str(sub.strip('\n'))
        line = line.split(',')
        res.update({line[0]: line[1]})
    fm.close()

    res['norm_length'] = float(res['norm_length'])
    settings.settings.norm_length = res['norm_length']
    res['discretisation'] = int(res['discretisation'])
    settings.settings.discretisation = res['discretisation']
    return res


def read_modalities_config(file_name):
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    res = []
    keys = [x for x in cfg['modalities']]
    for i in range(len(keys)):
        res.append([keys[i], cfg['modalities'][keys[i]]])
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
            res['use_constraint'] = True

        else:
            res['use_constraint'] = False
            # res.append( [keys[i],cfg['segmentation_conf'][keys[i]]])
        settings.settings.use_constraint = res['use_constraint']

    settings.settings.atlas_dir = res['atlas_dir']
    res['labels_to_segment'] = res['labels_to_segment'].split(',')
    settings.settings.labels_to_segment = res['labels_to_segment']
    if 'dependent_constraint' in res.keys():
        settings.settings.dependent_constraint = json.loads(res['dependent_constraint'])
    else:
        settings.settings.dependent_constraint = []

    if 'joint_labels' in res.keys():
        joint_labels_T = res['joint_labels'].split(',')
        for i in range(len(joint_labels_T)):
            joint_labels_T[i] = joint_labels_T[i].split('.')
        settings.settings.joint_labels = joint_labels_T
    else:
        settings.settings.joint_labels = None
    return res


def apply_transf_2_pts(pts, transf):
    res = []
    for i in range(len(pts)):
        pt_norm = np.array(pts[i] + [1])
        pt_tmp = list((np.dot(transf, pt_norm))[:3])

        res.append(pt_tmp)
    return res


def translate_p(p):
    """Creates translation matrix to p
    :return transormation matrix to py """
    m = np.eye(4)
    m[0, 3] = p[0]
    m[1, 3] = p[1]
    m[2, 3] = p[2]
    return m

def mirror_point(center, axis):
    p1 = translate_p(-center)
    p2 = translate_p(center)

    t = np.eye(4)
    t[axis,axis] = -1
    return np.linalg.multi_dot([p2,t,p1])

def rotation_along_axis_wc(center, axis,angle):
    center = np.array(center)
    p1 = translate_p(-center)
    p2 = translate_p(center)
    t = rotate_axis(axis, angle)
    return np.linalg.multi_dot([p2, t, p1])

def rotation_axis_2vecs(center, st_ax,rot_ax,N):
    center = np.array(center)
    p1 = translate_p(-center)
    p2 = translate_p(center)

    alpha = math.acos(np.dot(st_ax,rot_ax)/(np.linalg.norm(st_ax)*np.linalg.norm(rot_ax)))
    # if alpha > np.pi/2:
    #     alpha = 0 - (np.pi - alpha)
    t = rotate_axis(N,-alpha)
    return np.linalg.multi_dot([p2,t,p1])


def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]

def kron(a, b):
    a_s = []
    b_s = []
    if (a.shape == (3,)) & (b.shape == (3,)):
        a_s = [1, 3]
        b_s = [3, 1]

    A = np.reshape(a, (1, a_s[0], 1, a_s[1]))
    B = np.reshape(b, (b_s[0], 1, b_s[1], 1))
    K = np.reshape(A * B, [a_s[0] * a_s[1], b_s[0] * b_s[1]])
    return K


def rotate_axis(axis, degree):
    u = axis / np.linalg.norm(axis)
    m = np.eye(4)
    cosA = np.cos(degree)
    sinA = np.sin(degree)

    tmp = np.eye(4)
    kr = kron(u, u.transpose())
    tmp[0:3, 0:3] = cosA * np.eye(3) + (1 - cosA) * kr + sinA * np.array(
        [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]
    )
    m = np.dot(m, tmp)
    return m





def apply_transf_2_norms(norms, transf):
    a = np.array(norms)

    res = []
    for i in range(a.shape[0]):
        tmp = []
        for j in range(a.shape[1]):
            pt_norm = np.array(list(a[i, j, :]) + [1])
            pt_tmp = list((np.dot(transf, pt_norm))[:3])
            tmp.append(pt_tmp)
        res.append(tmp)
    return res


def read_fsl_mni2native_w(subject):
    im_mni = f_im.Image(subject + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")
    im_native = f_im.Image(subject + os.sep + "t1_acpc_extracted.nii.gz")

    fsl_mni2nat = fl.readFlirt(subject + os.sep + "combined_affine_reverse.mat")

    mni_w = fl.fromFlirt(fsl_mni2nat, im_mni, im_native, 'world', 'world')
    return mni_w


def read_fsl_native2mni_w(subject):
    a = read_fsl_mni2native_w(subject)
    return np.linalg.inv(a)


# method for compute position of label in data based on label
def comp_posit_in_data(label):
    return settings.settings.all_labels.index(label)


# generates point array
def generate_mask(x_r, y_r, z_r, dt):
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

    res_arr = np.zeros((len(x_arr), len(y_arr), len(z_arr), 3))
    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            for k in range(len(z_arr)):
                res_arr[i, j, k, 0] = x_arr[i]
                res_arr[i, j, k, 1] = y_arr[j]
                res_arr[i, j, k, 2] = z_arr[k]
    return res_arr


#########7p
def concatenate_intensities(x1, x2):
    res = []
    if x1 == []: return x2
    for i in range(len(x2)):
        res.append(x1[i] + x2[i])
    return res


def calc_intensities(norms, image):
    a = np.array(norms)
    res = []
    for i in range(a.shape[0]):
        tmp = []
        for j in range(a.shape[1]):
            pt_norm = list(a[i, j, :])
            intens = image.interpolate(pt_norm)
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


def calculate_mask_touches(mask, norms):
    res = []
    for i in range(len(norms)):
        tmp = []
        for j in range(len(norms[0])):
            tmp.append(mask.check_neighbours_world(voxel=norms[i][j]))
        res.append(tmp)
    return res

def apply_transform_to_point(transform, point):
    b = np.array(point)
    a = np.array(b.tolist() + [1])
    return np.dot(transform, a)[:3]

def save_intensities_csv(pdm, filename):
    try:
        os.remove(filename)
    except:
        pass
    f = open(filename, 'w')
    wr = csv.writer(f)
    wr.writerows(pdm)
    f.close()

def get_flirt_transformation_matrix(mat_file,src_file,dest_file,from_,to):
    im_src = fim.Image(src_file,loadData=False)
    im_dest = fim.Image(dest_file,loadData=False)
    forward_transf_fsl = fl.readFlirt(mat_file)
    return fl.fromFlirt(forward_transf_fsl,im_src,im_dest,from_,to)


def calculate_intensites_subject(modalities, labels, subject, discretisation, norm_len, mesh_name_end):
    images = []
    for i in range(len(modalities)):
        im = vtim.Image(subject + os.sep + modalities[i][1])
        im.setup_bspline(3)
        images.append([modalities[i][0], im])

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
        # calculate overlap of normal with mask in world coords
        mask_norm = calculate_mask_touches(mask=mask_im, norms=norms_native)
        norms_native = apply_transf_2_norms(norms_native, tr_World_voxel)
        surf.apply_transform(from_mni.tolist())
        # calc points
        mp = surf.generate_mesh_points(20)

        # calculate intensities mask,1st_modal,2nd...
        profiles = mask_norm
        means = []
        for j in range(len(images)):
            profile = calc_intensities(norms_native, images[j][1])
            mp2 = apply_transf_2_pts(mp, images[j][1]._world_2_vox)
            ######intensity blok(-mean)
            mn = np.array(images[j][1].interpolate_list(mp2)).mean()
            means.append(mn)
            # mn = 0 # for clean mean
            profiles = concatenate_intensities(profiles, (np.array(profile) - mn).tolist())
        norm_vecs = norms_2_coords(normals=mni_norms)
        means = means + [volum]
        # calc result mat
        res = concatenate_intensities(norm_vecs, profiles)
        save_intensities_csv(pdm=res + [means], filename=subject + os.sep + labels[i] + "_profiles.csv")


def points_2_fcsv(pts, filename):
    try:
        os.remove(path=filename)
    except:
        pass

    with open(filename, 'wt') as the_file:
        the_file.write("# Markups fiducial file version = 4.10\n")
        the_file.write("# CoordinateSystem = 0\n")
        the_file.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")

        for i in range(len(pts)):
            st = str(i) + ","
            st += str(pts[i][0]) + "," + str(pts[i][1]) + "," + str(pts[i][2])

            st += ",0.000,0.000,0.000,1.000,1,1,0," + ",,vtkMRMLScalarVolumeNode1\n"
            the_file.write(st)
