"""GENERATE NEW IMAGE FROM STATISTICS"""
import pickle

import bayessian_appearance.utils as util
import ExtPy
import numpy as np
import pandas as pd
import sys
import os
from pandas_ods_reader import read_ods
import sklearn.covariance as cov
import matplotlib.pyplot as plt
import scipy.io as io


def main_proc(inp, outp):
    # read im

    # meshes = np.array(meshes)
    m_mx = []


def calculate_statistics_anatomically(workdir, subjects):
    mesh_name_1 = "3_1T1.obj"
    mesh_name_2 = "4_1T1.obj"

    folder = workdir

    # points positions of intersection in T1 space
    l_pts_t1 = []
    r_pts_t1 = []

    # point position of intersectioned subjects
    l_pos_isected = []
    r_pos_isected = []
    for i in range(len(subjects)):
        r_mesh = ExtPy.cMesh(folder + os.sep + "sub-P"
                             + subjects[i] + os.sep + mesh_name_2)
        l_mesh = ExtPy.cMesh(folder + os.sep + "sub-P"
                             + subjects[i] + os.sep + mesh_name_1)
        r_et = [LEnt[i, :].tolist(), LTrg[i, :].tolist()]

        a = r_mesh.index_of_intersectedtriangle(r_et)  # right

        l_et = [REnt[i, :].tolist(), RTrg[i, :].tolist()]

        b = l_mesh.index_of_intersectedtriangle(l_et)  # left

        if not ((a[0]) == -1):
            r_pts_t1.append(a)
            r_pos_isected.append(subjs[i])

        if not ((b[0]) == -1):
            l_pts_t1.append(b)
            l_pos_isected.append(subjs[i])

    l_pts_t1 = np.array(l_pts_t1)
    r_pts_t1 = np.array(r_pts_t1)

    left_mesh = ExtPy.cMesh("/home/varga/processing_data/workdir/3_mean.obj")
    right_mesh = ExtPy.cMesh("/home/varga/processing_data/workdir/4_mean.obj")

    l_centers = np.array(left_mesh.centes_of_triangles())
    r_centers = np.array(right_mesh.centes_of_triangles())

    in_l_distr = cov.EllipticEnvelope(random_state=0)
    in_l_distr.fit(l_centers[l_pts_t1[:, 0]])

    out_l_distr = cov.EllipticEnvelope(random_state=0)
    out_l_distr.fit(l_centers[l_pts_t1[:, 1]])

    in_r_distr = cov.EllipticEnvelope(random_state=0)
    in_r_distr.fit(r_centers[r_pts_t1[:, 0]])

    out_r_distr = cov.EllipticEnvelope(random_state=0)
    out_r_distr.fit(r_centers[r_pts_t1[:, 1]])

    return [in_r_distr.mahalanobis(r_centers),
            out_r_distr.mahalanobis(r_centers),
            in_l_distr.mahalanobis(l_centers),
            out_l_distr.mahalanobis(l_centers),
            in_r_distr,
            out_r_distr,
            in_l_distr,
            out_l_distr

            ]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_file = sys.argv[1]

    df = read_ods(input_file, "surgical")

    sb = "subject"
    xLT = "TargetL1X"
    yLT = "TargetL1Y"
    zLT = "TargetL1Z"

    xLE = "EntryL1X"
    yLE = "EntryL1Y"
    zLE = "EntryL1Z"

    xRT = "TargetR1X"
    yRT = "TargetR1Y"
    zRT = "TargetR1Z"

    xRE = "EntryR1X"
    yRE = "EntryR1Y"
    zRE = "EntryR1Z"

    notNanSubjects = df[sb].values[np.where(~np.isnan(df[yLE]))].astype(np.int)

    xLEV = df[xLE].values[np.where(~np.isnan(df[yLE]))]
    yLEV = df[yLE].values[np.where(~np.isnan(df[yLE]))]

    LEnt = np.array(
        [df[xLE].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[yLE].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[zLE].values[np.where(~np.isnan(df[yLE]))].tolist()]
    ).transpose()
    REnt = np.array(
        [df[xRE].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[yRE].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[zRE].values[np.where(~np.isnan(df[yLE]))].tolist()]
    ).transpose()
    LTrg = np.array(
        [df[xLT].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[yLT].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[zLT].values[np.where(~np.isnan(df[yLE]))].tolist()]
    ).transpose()
    RTrg = np.array(
        [df[xRT].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[yRT].values[np.where(~np.isnan(df[yLE]))].tolist(),
         df[zRT].values[np.where(~np.isnan(df[yLE]))].tolist()]
    ).transpose()
    subjs = []
    for i in range(notNanSubjects.shape[0]):
        if notNanSubjects[i] < 100:
            subjs.append("0" + str(notNanSubjects[i]))
        else:
            subjs.append(str(notNanSubjects[i]))

    #######folder####################
    mesh_name_1 = "3_1T1.obj"
    mesh_name_2 = "4_1T1.obj"

    folder = sys.argv[2]

    [r_e, r_ex, l_e, l_ex,
     rc_1, rc_2, lc_1, lc_2] = calculate_statistics_anatomically(folder, subjs)
    l_mesh_ino = []
    r_mesh_ino = []

    l_mesh_centers = []
    r_mesh_centers = []

    # points positions of intersection in T1 space
    l_pts_t1 = []
    r_pts_t1 = []

    # point position of intersectioned subjects
    l_pos_isected = []
    r_pos_isected = []
    for i in range(len(subjs)):
        r_mesh = ExtPy.cMesh(folder + os.sep + "sub-P"
                             + subjs[i] + os.sep + mesh_name_2)
        l_mesh = ExtPy.cMesh(folder + os.sep + "sub-P"
                             + subjs[i] + os.sep + mesh_name_1)
        r_et = [LEnt[i, :].tolist(), LTrg[i, :].tolist()]

        a = r_mesh.ray_mesh_intersection(r_et)  # right

        l_et = [REnt[i, :].tolist(), RTrg[i, :].tolist()]

        b = l_mesh.ray_mesh_intersection(l_et)  # left

        tr_F = open(folder + os.sep + "sub-P"
                    + subjs[i] + os.sep + "transformACPC", "rb")
        transf = pickle.load(tr_F)
        from_ACPC = np.linalg.inv(transf)

        native_2_mni = util.read_fsl_native2mni_w(folder + os.sep + "sub-P"
                                                  + subjs[i] + os.sep)

        tr_F.close()

        l_centers = l_mesh.centes_of_triangles()
        r_centers = r_mesh.centes_of_triangles()

        l_centers = util.apply_transf_2_pts(l_centers, from_ACPC)
        l_centers = util.apply_transf_2_pts(l_centers, native_2_mni)
        r_centers = util.apply_transf_2_pts(r_centers, from_ACPC)
        r_centers = util.apply_transf_2_pts(r_centers, native_2_mni)

        # r
        a1 = util.apply_transform_to_point(from_ACPC, a[0])
        a2 = util.apply_transform_to_point(from_ACPC, a[1])
        # b
        b1 = util.apply_transform_to_point(from_ACPC, b[0])
        b2 = util.apply_transform_to_point(from_ACPC, b[1])

        a1 = util.apply_transform_to_point(native_2_mni, a1)
        a2 = util.apply_transform_to_point(native_2_mni, a2)

        b1 = util.apply_transform_to_point(native_2_mni, b1)
        b2 = util.apply_transform_to_point(native_2_mni, b2)

        if not ((min(a[0]) == 0) and (max(a[0]) == 0)):
            r_mesh_ino.append([a1, a2])
            r_mesh_centers.append(r_centers)
            r_pts_t1.append(a)
            r_pos_isected.append(subjs[i])

        if not ((min(b[0]) == 0) and (max(b[0]) == 0)):
            l_mesh_ino.append([b1, b2])
            l_mesh_centers.append(l_centers)
            l_pts_t1.append(b)
            l_pos_isected.append(subjs[i])

    l_mesh_ino = np.array(l_mesh_ino)
    r_mesh_ino = np.array(r_mesh_ino)

    # rc_1 = cov.EllipticEnvelope(random_state=0)
    # rc_1.fit(r_mesh_ino[:,0,:])
    #
    # rc_2 = cov.EllipticEnvelope(random_state=0)
    # rc_2.fit(r_mesh_ino[:, 1, :])
    #
    # rc_comb = cov.EllipticEnvelope(random_state=0)
    # rc_comb.fit(np.concatenate((r_mesh_ino[:,0,:],r_mesh_ino[:,1,:]),axis=-1))
    #
    #
    # lc_1 = cov.EllipticEnvelope(random_state=0)
    # lc_1.fit(l_mesh_ino[:,0,:])
    #
    # lc_2 = cov.EllipticEnvelope(random_state=0)
    # lc_2.fit(l_mesh_ino[:, 1, :])

    index_median_rght = np.where(rc_1.dist_ == min(rc_1.dist_))[0][0]
    lc_comb = cov.EllipticEnvelope()
    lc_comb.fit(np.concatenate((l_mesh_ino[:, 0, :], l_mesh_ino[:, 1, :]), axis=-1))
    index_median_lft = np.where(lc_1.dist_ == min(lc_1.dist_))[0][0]

    # get_median sub

    l_mesh_centers = np.array(l_mesh_centers)
    r_mesh_centers = np.array(r_mesh_centers)

    v1 = []
    v2 = []
    v3 = []
    v4 = []
    for i in range(len(l_centers)):
        entrL = lc_1.mahalanobis(l_mesh_centers[:, i, :])
        extL = lc_2.mahalanobis(l_mesh_centers[:, i, :])
        v1.append(np.median(entrL))
        v2.append(np.median(extL))

        entrR = rc_1.mahalanobis(r_mesh_centers[:, i, :])
        extR = rc_2.mahalanobis(r_mesh_centers[:, i, :])
        v4.append(np.median(entrR))
        v3.append(np.median(extR))

        print(1)

    v3 = np.array(v3)
    v4 = np.array(v4)

    v1 = np.array(v1)
    v2 = np.array(v2)
    # plt.figure()

    left = []
    right = []

    left_median_subj_name = l_pos_isected[index_median_lft]
    right_median_subj_name = r_pos_isected[index_median_rght]

    right_vecs_t1 = np.array(r_pts_t1[index_median_rght])
    left_vecs_t1 = np.array(l_pts_t1[index_median_lft])
    # r_e =np.log2(1+ (r_e - min(r_e))/(max(r_e) - min(r_e)))
    # r_ex =np.log2(1+ (r_ex - min(r_ex)) / (max(r_ex) - min(r_ex)))
    # l_e =np.log2(1+ (l_e - min(l_e)) / (max(l_e) - min(l_e)))
    # l_ex =np.log2(1+ (l_ex - min(l_ex)) / (max(l_ex) - min(l_ex)))
    for i in range(len(l_centers)):
        right.append(min([r_e[i], r_ex[i]]))
        left.append(min([l_e[i], l_ex[i]]))

    ### calc left in t1 space
    mni_2_nat = util.read_fsl_mni2native_w(folder + os.sep + "sub-P"
                                           + left_median_subj_name)
    left_native_in_out = []
    for i in range(l_mesh_ino.shape[0]):
        input_mni = l_mesh_ino[i, 0, :]
        inp_nat = util.apply_transform_to_point(mni_2_nat, input_mni)

        output_mni = l_mesh_ino[i, 1, :]
        out_nat = util.apply_transform_to_point(mni_2_nat, output_mni)

        left_native_in_out.append([inp_nat, out_nat])

    dc = {"left": np.array(left), "right": np.array(right),
          "right_name": right_median_subj_name, "left_name": left_median_subj_name,
          "right_position": right_vecs_t1, "left_position": left_vecs_t1,
          "left_ent": l_e, "left_ext": l_ex, "left_pts_in_out": np.array(left_native_in_out)}

    out_file = sys.argv[3]
    try:
        os.remove(out_file)
    except:
        pass
    io.savemat(out_file, dc)
