# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc_opt
import scipy.stats as sc_stat
import bayessian_appearance.utils as uti
import ExtPy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import scipy.optimize as opt
import matplotlib.tri as mtri
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# safe_subjects = [60,62,63,66,69,70,71,73,74,75,77,78,79,81,82,83,84,85,86,87,90,93,95,97,98,99,100,102,104,105,106,107,110,111,112,113,114,115,116,118,120,125,126,129,132,133]
safe_subjects = [60, 62, 63, 66, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 90, 93, 95, 97, 98, 99,
                 102, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 116, 118, 120, 125, 126, 129, 132, 133]


# safe_subjects = [60]

def read_all_subjects():
    f = open("/home/varga/mer_data_processing/test", 'rb')

    dat = pickle.load(f)
    return dat


def generate_mask_for_subject(subj_ind, subj, side, anat_labels):
    subs = np.array(anat_labels[0])

    ind = np.where(subs == subj_ind)[0][0]

    res_mask = []
    start_end = []
    for i in range(subj.get_num_electrodes()):
        el_name = subj.get_electrode_name_by_index(i)
        top = anat_labels[1][side][el_name]['top'][ind]
        bot = anat_labels[1][side][el_name]['bot'][ind]
        # calc mask

        a = np.array(subj.distances)

        if top == np.nan:
            a[:] = False
            res_mask.append(a)
            continue
        r_tmp = (a >= top) & (a <= bot)
        res_mask.append(r_tmp)

    return res_mask


def combine_into_one_array(parsed_data):
    comb_arr = parsed_data['right'] + parsed_data['left']

    comb_res = None
    for i in range(len(comb_arr)):
        if not (comb_arr[i] is None):
            for j in range(comb_arr[i].shape[0]):

                if comb_res is None:
                    comb_res = comb_arr[i][j]
                else:
                    if not (comb_arr[i] is None):
                        comb_res = np.concatenate((comb_res, comb_arr[i][j]))

    comb_arr = parsed_data['right_masks'] + parsed_data['left_masks']
    comb_masks = None
    for i in range(len(comb_arr)):
        if not (comb_arr[i] is None):
            ca = np.array(comb_arr[i])
            for j in range(ca.shape[0]):

                if comb_masks is None:
                    comb_masks = comb_arr[i][j]
                else:
                    if not (comb_arr[i] is None):
                        comb_masks = np.concatenate((comb_masks, comb_arr[i][j]))

    return [comb_res, comb_masks]


def train_lognorms(data):
    """:return IN and OUT distribution"""
    stns = data[0][data[1]]
    # stns = sc_stat.lognorm.rvs(s=0.5,loc=1,size=10000)
    n_stns = data[0][~ data[1]]
    ax = plt.subplot(111)
    ax.hist(stns, 50, density=True, facecolor='r', alpha=0.5)
    ax.hist(n_stns, 50, density=True, facecolor='b', alpha=0.5)
    # ax.set_xscale("log")

    shape, loc, scale = sc_stat.lognorm.fit(stns, loc=stns.mean())
    shape1, loc1, scale1 = sc_stat.lognorm.fit(n_stns, loc=n_stns.mean())

    x = np.logspace(0, 5, 200)
    pdf = sc_stat.lognorm.pdf(x, shape, loc, scale)
    pdf2 = sc_stat.lognorm.pdf(x, shape1, loc1, scale1)
    #
    plt.xlim(min(stns) - 2, max(stns) + 2)
    ax.plot(x, pdf, 'r')
    ax.plot(x, pdf2, 'b')
    pdf = lambda x : sc_stat.lognorm.pdf(x, shape, loc, scale)
    pdf2 = lambda x: sc_stat.lognorm.pdf(x, shape1, loc1, scale1)
    return [pdf, pdf2]


def compute_distance(map_inst, nrms, mask, distances):
    nrms = nrms.tolist()
    mask = mask.tolist()
    distances = distances

    ind_inside = mask.index(True)
    try:
        ind_outside = mask.index(False, ind_inside)
    except:
        ind_outside = len(mask)
    dist_copy = [x - distances[ind_inside] for x in distances]

    for i in range(ind_outside):
        if dist_copy[i] in map_inst:
            map_inst[dist_copy[i]].append(nrms[i])
        else:
            map_inst[dist_copy[i]] = [nrms[i]]

    ind_inside = mask.index(True)
    try:
        ind_outside = mask.index(False, ind_inside)
    except:
        ind_outside = len(mask)
    dist_copy = [x - distances[ind_outside - 1] for x in distances]

    for i in range(ind_inside, len(mask)):
        if -dist_copy[i] in map_inst:
            map_inst[-dist_copy[i]].append(nrms[i])
        else:
            map_inst[-dist_copy[i]] = [nrms[i]]

    # mask.reverse()
    # nrms.reverse()
    # distances.reverse()
    # distances = [ -x for x in distances]
    # ind_inside = mask.index(True)
    # ind_outside = mask.index(False, ind_inside)
    # dist_copy = [ x - distances[ind_inside] for x in distances]
    # for i in range(ind_outside):
    #     if dist_copy[i] in map_inst:
    #         map_inst[dist_copy[i]].append(nrms[i])
    #     else:
    #         map_inst[dist_copy[i]] = [ nrms[i]]


def read_edf_entry_target(ods_file):
    from pandas_ods_reader import read_ods
    df = read_ods(ods_file, "surgical")
    sb = "subject"

    TLX = "TargetL1X"
    TLY = "TargetL1Y"
    TLZ = "TargetL1Z"

    ELX = "EntryL1X"
    ELY = "EntryL1Y"
    ELZ = "EntryL1Z"

    TRX = "TargetR1X"
    TRY = "TargetR1Y"
    TRZ = "TargetR1Z"

    ERX = "EntryR1X"
    ERY = "EntryR1Y"
    ERZ = "EntryR1Z"

    ###get subjects
    subs = df[sb].values.astype(int).tolist()
    ###get su

    RS = {
        "entry": {"x": df[ELX].values, "y": df[ELY].values, "z": df[ELZ].values},
        "target": {"x": df[TLX].values, "y": df[TLY].values, "z": df[TLZ].values}
    }
    LS = {
        "entry": {"x": df[ERX].values, "y": df[ERY].values, "z": df[ERZ].values},
        "target": {"x": df[TRX].values, "y": df[TRY].values, "z": df[TRZ].values}
    }

    for el in RS:

        for a in RS[el]:
            RS[el][a][np.where(RS[el][a] == "n/a")] = np.nan
            RS[el][a][np.where(RS[el][a] == "nil")] = np.nan
            RS[el][a][np.where(RS[el][a] is None)] = np.nan

    for el in LS:

        for a in LS[el]:
            LS[el][a][np.where(LS[el][a] == "n/a")] = np.nan
            LS[el][a][np.where(LS[el][a] == "nil")] = np.nan
            LS[el][a][np.where(LS[el][a] is None)] = np.nan

    return [subs, {"left": LS, "right": RS}]


def func(x, a, b):
    return np.power((1 + np.exp(-a * x)), -b)


def train_sigmoid(data):
    rd = data['right_distances']

    ld = data['left_distances']

    comb_dist = data['right_distances'] + data['left_distances']
    comb_nrms = data['right'] + data['left']
    comb_masks = data['right_masks'] + data['left_masks']

    res_dists = {}
    for ind in range(len(comb_nrms)):

        for e_ind in range(comb_nrms[ind].shape[0]):
            masks = comb_masks[ind][e_ind]
            if not masks.max():
                continue
            compute_distance(res_dists, comb_nrms[ind][e_ind], masks, comb_dist[ind])
    func = lambda x, a, b: 1 / (1 + np.exp(-(b + a * x)))

    ks = []
    vals = []
    for key in res_dists.keys():
        ks = ks + [key] * len(res_dists[key])
        # ks.append(key)
        vals += res_dists[key]

    def sigmoid(x, x0, k, b):
        y = 1 / (1 + np.exp(-k * (x - x0))) + b
        return (y)

    p0 = [np.median(ks), 1, min(vals)]

    parameters, _ = sc_opt.curve_fit(func, xdata=ks, ydata=vals, method="lm"
                                     # ,bounds=([-math.inf,0 ],[math.inf,math.inf] )
                                     )
    popt, _ = sc_opt.curve_fit(sigmoid, ks, vals, p0, method='dogbox')
    # parameters = sc_opt.least_squares(func,xdata=ks,ydata=vals,method="ls"
    #                               #,bounds=([-math.inf,0 ],[math.inf,math.inf] )
    #                               )

    #f_res = lambda x: func(x,parameters[0],parameters[1])
    f_res = lambda x: sigmoid(x, popt[0], popt[1], popt[2])
    #f_res = lambda x: func(x, parameters[0][0])
    min_k = min(res_dists.keys())
    max_k = max(res_dists.keys())

    plt.figure()
    plt.scatter(ks, vals, alpha=0.01)
    f_res = lambda x: sigmoid(x, popt[0], popt[1], 0)
    xdata = np.linspace(min_k, max_k, 100)
    ydata = f_res(xdata)
    plt.plot(xdata, ydata, 'r')
    plt.show()

    return f_res


def parse_lengths(subs, trajectories):
    lefts = []
    rights = []
    for ss in safe_subjects:
        ind = subs.index(ss)
        coords_le = [trajectories['left']['entry']['x'][ind], trajectories['left']['entry']['y'][ind],
                     trajectories['left']['entry']['z'][ind]]
        coords_lt = [trajectories['left']['target']['x'][ind], trajectories['left']['target']['y'][ind],
                     trajectories['left']['target']['z'][ind]]

        coords_re = [trajectories['right']['entry']['x'][ind], trajectories['right']['entry']['y'][ind],
                     trajectories['right']['entry']['z'][ind]]
        coords_rt = [trajectories['right']['target']['x'][ind], trajectories['right']['target']['y'][ind],
                     trajectories['right']['target']['z'][ind]]

        left = [coords_le, coords_lt]
        lefts.append(left)

        right = [coords_re, coords_rt]
        rights.append(right)
    return {"left" : lefts, "right": rights}


def plot_3d_plane(points,points2):


    p0, p1, p2 = points
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
    vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

    u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

    point = np.array(p0)
    normal = np.array(u_cross_v)

    d = -point.dot(normal)
    if normal[2] == 0 and normal[2] == 0 and normal[1] != 0:
        xx, z = np.meshgrid(range(100), range(100))
        yy = xx*0 + (-d)/normal[1]
        pass
    else:
        xx, yy = np.meshgrid(range(100), range(100))


        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z,alpha=0.2)

    p0, p1, p2 = points2
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
    vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

    u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

    point = np.array(p0)
    normal = np.array(u_cross_v)

    d = -point.dot(normal)

    xx, yy = np.meshgrid(range(100), range(100))

    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    plt3d = plt.gca(projection='3d')
    plt3d.plot_surface(xx, yy, z,alpha=0.2)

    plt.show()
def pts_plot(toacpc,left,right,el_names_r,el_names_l):

    toacpc = np.array(toacpc)
    from_acpc = np.linalg.inv(toacpc)


    rght = uti.apply_transf_2_pts(right,toacpc)
    lft = uti.apply_transf_2_pts(left, toacpc)

    lft_t = uti.apply_transf_2_pts(lft,from_acpc)

    result_r = {}
    result_l = {}
    cen_l = [0,0,0]
    lat_l = [2,0,0]
    med_l = [2,0,0]
    ant_l = [0,2,0]
    pos_l = [0,-2,0]

    cen_l2 = [0,0,-1]
    lat_l2 = [2,0,-1]
    med_l2 = [2,0,-1]
    ant_l2 = [0,2,-1]
    pos_l2 = [0,-2,-1]

    cen_r = [0, 0, 0]
    lat_r = [-2, 0, 0]
    med_r = [2, 0, 0]
    ant_r = [0, 2, 0]
    pos_r = [0, -2, 0]

    cen_r2 = [0, 0, -1 ]
    lat_r2 = [-2, 0, -1]
    med_r2 = [2, 0, -1 ]
    ant_r2 = [0, 2, -1 ]
    pos_r2 = [0, -2, -1]
    ##################### LEFT ##########################

    coef= np.linalg.norm(np.array(lft[1]) - np.array(lft[0]))
    cen_l2[2] = cen_l2[2] * coef
    lat_l2[2] = lat_l2[2] * coef
    med_l2[2] = med_l2[2] * coef
    ant_l2[2] = ant_l2[2] * coef
    pos_l2[2] = pos_l2[2] * coef

    match_pts = uti.translate_p(np.array(lft[0]))

    # [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
    #     [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=match_pts)

    c1 = np.array(cen_l)
    c2 = np.array(cen_l2)
    c3 = np.array(med_l)

    a1 = c1 - c2
    a2 = c3 - c2
    a1 = a1[0:3]
    a2 = a2[0:3]
    a_cross = np.cross(a1, a2)
    a_cross = a_cross/np.linalg.norm(a_cross)

    z1 = np.array(lft[0])
    z2 = np.array(lft[1])
    z3 = np.array(lft[0]) + np.array([2,0,0])
    #plot_3d_plane([c1,c2,c3],[z1,z2,z3])

    mni_a1 = z1 - z2
    mni_a2 = z3 - z2
    mni_across = np.cross(mni_a1, mni_a2)
    mni_across = mni_across/np.linalg.norm(mni_across)

    d1 = -np.dot(a_cross, c1)
    d2 = -np.dot(mni_across, z1)
    res_intr = uti.plane_intersect(list(a_cross) + [d1], list(mni_across) + [d2])
    p = res_intr[0]
    N = (res_intr[1] - res_intr[0]) / np.linalg.norm(res_intr[1] - res_intr[0])
    #angle between two planes (MNI coords and MNI origin PLANE)
    alpha = np.arccos(np.dot(mni_across, a_cross) / (np.linalg.norm(mni_across) * np.linalg.norm(a_cross)))
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)
    print(alpha)

    rotA = uti.rotate_axis(N, alpha)

    c1,c2,c3 = uti.apply_transf_2_pts([c1.tolist(),c2.tolist(),c3.tolist()],rotA)
    match_pts = uti.translate_p(np.array(lft[0])-np.array(c1))
    alpha = np.arccos(np.dot(np.array(c1)-np.array(c2), np.array(z1)-np.array(z2))
                      / (np.linalg.norm(np.array(c1)-np.array(c2)) * np.linalg.norm(np.array(z1)-np.array(z2))))
    rotB = uti.rotate_axis(mni_across, alpha)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], rotB)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], match_pts)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=rotA)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=rotB)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=match_pts)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=from_acpc)
    if 'central' in el_names_l:
        result_l['central'] = [cen_l,cen_l2]
    if 'anterior' in el_names_l:
        result_l['anterior'] = [ant_l, ant_l2]
    if 'posterior' in el_names_l:
        result_l['posterior'] = [pos_l,pos_l2]
    if 'lateral' in el_names_l:
        result_l['lateral'] = [lat_l,lat_l2]
    if 'medial' in el_names_l:
        result_l['medial'] = [med_l,med_l2]


    ###################################################RIGHT

    coef= np.linalg.norm(np.array(rght[1]) - np.array(rght[0]))
    cen_r2[2] = cen_r2[2] * coef
    lat_r2[2] = lat_r2[2] * coef
    med_r2[2] = med_r2[2] * coef
    ant_r2[2] = ant_r2[2] * coef
    pos_r2[2] = pos_r2[2] * coef

    #match_pts = uti.translate_p(np.array(lft[0]))

    # [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
    #     [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=match_pts)

    c1 = np.array(cen_r)
    c2 = np.array(cen_r2)
    c3 = np.array(med_r)

    a1 = c1 - c2
    a2 = c3 - c2
    a1 = a1[0:3]
    a2 = a2[0:3]
    a_cross = np.cross(a1, a2)
    a_cross = a_cross/np.linalg.norm(a_cross)

    z1 = np.array(rght[0])
    z2 = np.array(rght[1])
    z3 = np.array(rght[0]) + np.array([-2,0,0])
    #plot_3d_lines([c1, c2, c3],[ z1, z2, z3])
    #plot_3d_plane([c1, c2, c3], [z1, z2, z3])

    mni_a1 = z1 - z2
    mni_a2 = z3 - z2
    mni_across = np.cross(mni_a1, mni_a2)
    mni_across = mni_across/np.linalg.norm(mni_across)

    d1 = -np.dot(a_cross, c1)
    d2 = -np.dot(mni_across, z1)
    res_intr = uti.plane_intersect(list(a_cross) + [d1], list(mni_across) + [d2])
    p = res_intr[0]
    N = (res_intr[1] - res_intr[0]) / np.linalg.norm(res_intr[1] - res_intr[0])
    #angle between two planes (MNI coords and MNI origin PLANE)
    alpha = np.arccos(np.dot(mni_across, a_cross) / (np.linalg.norm(mni_across) * np.linalg.norm(a_cross)))
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)
    print(alpha)

    rotA = uti.rotate_axis(N, alpha)

    c1,c2,c3 = uti.apply_transf_2_pts([c1.tolist(),c2.tolist(),c3.tolist()],rotA)
    match_pts = uti.translate_p(np.array(rght[0])-np.array(c1))
    alpha = np.arccos(np.dot(np.array(c1)-np.array(c2), np.array(z1)-np.array(z2))
                      / (np.linalg.norm(np.array(c1)-np.array(c2)) * np.linalg.norm(np.array(z1)-np.array(z2))))
    rotB = uti.rotate_axis(mni_across, alpha)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], rotB)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], match_pts)
    #plot_3d_lines(np.array([c1, c2, c3]),np.array([ z1, z2, z3]))
    #plot_3d_plane([c1, c2, c3], [z1, z2, z3])
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])
    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=rotA)
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])

    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=rotB)
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])

    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=match_pts)
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])

    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=from_acpc)
    if 'central' in el_names_r:
        result_r['central'] = [cen_r,cen_r2]
    if 'anterior' in el_names_r:
        result_r['anterior'] = [ant_r, ant_r2]
    if 'posterior' in el_names_r:
        result_r['posterior'] = [pos_r,pos_r2]
    if 'lateral' in el_names_r:
        result_r['lateral'] = [lat_r,lat_r2]
    if 'medial' in el_names_r:
        result_r['medial'] = [med_r,med_r2]


    ##ret
    return {'left': result_l,'right' : result_r}
    #caclulate intersection of data MNI coords and MNI origin PLANE


    #################### RR ###########################


def generate_lines(toacpc,left,right,el_names_r,el_names_l):

    toacpc = np.array(toacpc)
    from_acpc = np.linalg.inv(toacpc)


    rght = uti.apply_transf_2_pts(right,toacpc)
    lft = uti.apply_transf_2_pts(left, toacpc)

    lft_t = uti.apply_transf_2_pts(lft,from_acpc)

    result_r = {}
    result_l = {}
    cen_l = [0,0,0]
    lat_l = [2,0,0]
    med_l = [2,0,0]
    ant_l = [0,2,0]
    pos_l = [0,-2,0]

    cen_l2 = [0,0,-1]
    lat_l2 = [2,0,-1]
    med_l2 = [2,0,-1]
    ant_l2 = [0,2,-1]
    pos_l2 = [0,-2,-1]

    cen_r = [0, 0, 0]
    lat_r = [-2, 0, 0]
    med_r = [2, 0, 0]
    ant_r = [0, 2, 0]
    pos_r = [0, -2, 0]

    cen_r2 = [0, 0, -1 ]
    lat_r2 = [-2, 0, -1]
    med_r2 = [2, 0, -1 ]
    ant_r2 = [0, 2, -1 ]
    pos_r2 = [0, -2, -1]
    ##################### LEFT ##########################

    coef= np.linalg.norm(np.array(lft[1]) - np.array(lft[0]))
    cen_l2[2] = cen_l2[2] * coef
    lat_l2[2] = lat_l2[2] * coef
    med_l2[2] = med_l2[2] * coef
    ant_l2[2] = ant_l2[2] * coef
    pos_l2[2] = pos_l2[2] * coef

    match_pts = uti.translate_p(np.array(lft[0]))

    # [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
    #     [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=match_pts)

    c1 = np.array(cen_l)
    c2 = np.array(cen_l2)
    c3 = np.array(med_l)

    a1 = c1 - c2
    a2 = c3 - c2
    a1 = a1[0:3]
    a2 = a2[0:3]
    a_cross = np.cross(a1, a2)
    a_cross = a_cross/np.linalg.norm(a_cross)

    z1 = np.array(lft[0])
    z2 = np.array(lft[1])
    z3 = np.array(lft[0]) + np.array([2,0,0])
    #plot_3d_plane([c1,c2,c3],[z1,z2,z3])

    mni_a1 = z1 - z2
    mni_a2 = z3 - z2
    mni_across = np.cross(mni_a1, mni_a2)
    mni_across = mni_across/np.linalg.norm(mni_across)

    d1 = -np.dot(a_cross, c1)
    d2 = -np.dot(mni_across, z1)
    res_intr = uti.plane_intersect(list(a_cross) + [d1], list(mni_across) + [d2])
    p = res_intr[0]
    N = (res_intr[1] - res_intr[0]) / np.linalg.norm(res_intr[1] - res_intr[0])
    #angle between two planes (MNI coords and MNI origin PLANE)
    alpha = np.arccos(np.dot(mni_across, a_cross) / (np.linalg.norm(mni_across) * np.linalg.norm(a_cross)))
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)
    print(alpha)

    rotA = uti.rotate_axis(N, alpha)

    c1,c2,c3 = uti.apply_transf_2_pts([c1.tolist(),c2.tolist(),c3.tolist()],rotA)
    match_pts = uti.translate_p(np.array(lft[0])-np.array(c1))
    alpha = np.arccos(np.dot(np.array(c1)-np.array(c2), np.array(z1)-np.array(z2))
                      / (np.linalg.norm(np.array(c1)-np.array(c2)) * np.linalg.norm(np.array(z1)-np.array(z2))))
    rotB = uti.rotate_axis(mni_across, alpha)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], rotB)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], match_pts)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=rotA)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=rotB)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=match_pts)
    [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
        [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=from_acpc)
    if 'central' in el_names_l:
        result_l['central'] = [cen_l,cen_l2]
    if 'anterior' in el_names_l:
        result_l['anterior'] = [ant_l, ant_l2]
    if 'posterior' in el_names_l:
        result_l['posterior'] = [pos_l,pos_l2]
    if 'lateral' in el_names_l:
        result_l['lateral'] = [lat_l,lat_l2]
    if 'medial' in el_names_l:
        result_l['medial'] = [med_l,med_l2]


    ###################################################RIGHT

    coef= np.linalg.norm(np.array(rght[1]) - np.array(rght[0]))
    cen_r2[2] = cen_r2[2] * coef
    lat_r2[2] = lat_r2[2] * coef
    med_r2[2] = med_r2[2] * coef
    ant_r2[2] = ant_r2[2] * coef
    pos_r2[2] = pos_r2[2] * coef

    #match_pts = uti.translate_p(np.array(lft[0]))

    # [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l] = uti.apply_transf_2_pts(
    #     [cen_l,cen_l2 , lat_l2, med_l2, ant_l2,pos_l2,cen_l,lat_l,med_l,ant_l,pos_l],transf=match_pts)

    c1 = np.array(cen_r)
    c2 = np.array(cen_r2)
    c3 = np.array(med_r)

    a1 = c1 - c2
    a2 = c3 - c2
    a1 = a1[0:3]
    a2 = a2[0:3]
    a_cross = np.cross(a1, a2)
    a_cross = a_cross/np.linalg.norm(a_cross)

    z1 = np.array(rght[0])
    z2 = np.array(rght[1])
    z3 = np.array(rght[0]) + np.array([-2,0,0])
    #plot_3d_lines([c1, c2, c3],[ z1, z2, z3])
    #plot_3d_plane([c1, c2, c3], [z1, z2, z3])

    mni_a1 = z1 - z2
    mni_a2 = z3 - z2
    mni_across = np.cross(mni_a1, mni_a2)
    mni_across = mni_across/np.linalg.norm(mni_across)

    d1 = -np.dot(a_cross, c1)
    d2 = -np.dot(mni_across, z1)
    res_intr = uti.plane_intersect(list(a_cross) + [d1], list(mni_across) + [d2])
    p = res_intr[0]
    N = (res_intr[1] - res_intr[0]) / np.linalg.norm(res_intr[1] - res_intr[0])
    #angle between two planes (MNI coords and MNI origin PLANE)
    alpha = np.arccos(np.dot(mni_across, a_cross) / (np.linalg.norm(mni_across) * np.linalg.norm(a_cross)))
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)
    print(alpha)

    rotA = uti.rotate_axis(N, alpha)

    c1,c2,c3 = uti.apply_transf_2_pts([c1.tolist(),c2.tolist(),c3.tolist()],rotA)
    match_pts = uti.translate_p(np.array(rght[0])-np.array(c1))
    alpha = np.arccos(np.dot(np.array(c1)-np.array(c2), np.array(z1)-np.array(z2))
                      / (np.linalg.norm(np.array(c1)-np.array(c2)) * np.linalg.norm(np.array(z1)-np.array(z2))))
    rotB = uti.rotate_axis(mni_across, alpha)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], rotB)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], match_pts)
    #plot_3d_lines(np.array([c1, c2, c3]),np.array([ z1, z2, z3]))
    #plot_3d_plane([c1, c2, c3], [z1, z2, z3])
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])
    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=rotA)
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])

    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=rotB)
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])

    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=match_pts)
    # plot_lines_list([[cen_r, cen_r2],
    #                  [lat_r, lat_r2],
    #                  [ant_r, ant_r2],
    #                  [pos_r, pos_r2],
    #                  [med_r, med_r2]])

    [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r] = uti.apply_transf_2_pts(
        [cen_r,cen_r2 , lat_r2, med_r2, ant_r2,pos_r2,cen_r,lat_r,med_r,ant_r,pos_r],transf=from_acpc)
    if 'central' in el_names_r:
        result_r['central'] = [cen_r,cen_r2]
    if 'anterior' in el_names_r:
        result_r['anterior'] = [ant_r, ant_r2]
    if 'posterior' in el_names_r:
        result_r['posterior'] = [pos_r,pos_r2]
    if 'lateral' in el_names_r:
        result_r['lateral'] = [lat_r,lat_r2]
    if 'medial' in el_names_r:
        result_r['medial'] = [med_r,med_r2]


    ##ret
    return {'left': result_l,'right' : result_r}
    #caclulate intersection of data MNI coords and MNI origin PLANE


    #################### RR ###########################

def plot_3d_lines(tringle1,triangle2):
    plt3d = plt.figure().gca(projection='3d')

    ln1 = np.array([tringle1[0].tolist(),tringle1[1].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2])
    ln1 = np.array([tringle1[0].tolist(),tringle1[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2])
    ln1 = np.array([tringle1[1].tolist(),tringle1[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2])

    ln1 = np.array([triangle2[0].tolist(),triangle2[1].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2])
    ln1 = np.array([triangle2[0].tolist(),triangle2[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2])
    ln1 = np.array([triangle2[1].tolist(),triangle2[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2])

    plt.show()


def plot_electrodes(initialised,meshes):
    plt3d = plt.figure().gca(projection='3d')

    for mesh in meshes:
        fc = np.array(mesh.get_faces())
        v = mesh.get_unpacked_coords()
        v = np.array(v)
        v = v.reshape((int(v.shape[0] / 3), 3))
        # cluster to set
        plt3d.scatter3D(v[0][0],v[0][1],v[0][2])
        plt3d.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=fc)

       # plt3d.add_collection3d(pc)
    for key in initialised.keys():

        for k2 in initialised[key].keys():
            pts = np.array(initialised[key][k2])
            plt3d.plot(pts[:,0],pts[:,1],pts[:,2])
    plt.show()



def plot_lines_list(lines):
    plt3d = plt.figure().gca(projection='3d')

    # for mesh in meshes:
    #     fc = np.array(mesh.get_faces())
    #     v = mesh.get_unpacked_coords()
    #     v = np.array(v)
    #     v = v.reshape((int(v.shape[0] / 3), 3))
    #     # cluster to set
    #
    #     pc = art3d.Poly3DCollection(v[fc])
    #     plt3d.add_collection3d(pc)
    for key in lines:
        pts = np.array(key)
        plt3d.plot(pts[:,0],pts[:,1],pts[:,2])
    # for key in initialised[1].keys():
    #     pts = np.array(initialised[1][key])
    #     plt3d.plot(pts[:,0],pts[:,1],pts[:,2])
    plt.show()



def compute_lognorm(distance,nrms,pdfs,sigmoid):

    distance = -distance
    psig = sigmoid(distance)
    return 100 * (- math.log10( (1- psig)*pdfs[1](nrms) + psig*pdfs[0](nrms)))

def compute_position_for_line(line,distances):
    dir_vect = np.array(line[0]) - np.array(line[1])
    dir_vect = dir_vect/np.linalg.norm(dir_vect)

    p0 = np.array(line[1])

    return np.array( [ p0 + t*(-dir_vect) for t in distances])



def displace_points_along_electrodes(el_name,distances,lines):
    res_lines = [ ]
    for key in el_name:
        line = lines[key]
        t_dist = compute_position_for_line(line,distances)
        res_lines.append(t_dist)
    return res_lines


def fit_side(displaced_electrodes,subdata, mesh, sigmoid,pdf):


    def functional(displaced_electrodes,subdata,mesh,sigmoid,pdf,transformation ):

        tfm = uti.translate_p(transformation)#compute only translations
        mesh.apply_transform(tfm)
        r_side_vals = []
        for i in range(subdata.shape[0]):
            for j in range(subdata.shape[1]):
                nrms = subdata[i, j]
                displ_position = displaced_electrodes[i][j]
                distance = mesh.distance_to_point(displ_position[0], displ_position[1], displ_position[2])
                r_side_vals.append(compute_lognorm(distance, nrms, pdf, sigmoid))
        tfm = uti.translate_p(-transformation)
        mesh.apply_transform(tfm)
        logn_value = np.mean(r_side_vals)

        return logn_value

    fc = lambda x: functional(displaced_electrodes,subdata,mesh,sigmoid,pdf,x)

    fm  = opt.minimize(fc, x0=np.array([0,0,0]),method="Powell")
    tfm = uti.translate_p(fm.x)
    mesh.apply_transform(tfm)
    return mesh




def plot_electrode_as_pts(displace_along_els,subdata,mesh,mask):
    plt3d = plt.figure().gca(projection='3d')


    fc = np.array(mesh.get_faces())
    v = mesh.get_unpacked_coords()
    v = np.array(v)
    v = v.reshape((int(v.shape[0] / 3), 3))
    # cluster to set
    plt3d.scatter3D(v[0][0],v[0][1],v[0][2])
    plt3d.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=fc,alpha=0.3)
    for i in range(subdata.shape[0]):
        sbd = subdata[i]
        r_msk = mask[i]
        sbd = 5*((sbd - min(sbd) )/(max(sbd)- min(sbd))) + 1
        plt3d.scatter(displace_along_els[i][r_msk, 0]
                      , displace_along_els[i][r_msk, 1],
                      displace_along_els[i][r_msk, 2], s=sbd[r_msk],c='g')

        plt3d.scatter(displace_along_els[i][~r_msk,0]
                      ,displace_along_els[i][~r_msk,1],
                      displace_along_els[i][~r_msk,2],s=sbd[~r_msk],c='r')
       # plt3d.add_collection3d(pc)
    # for key in initialised.keys():
    #
    #     for k2 in initialised[key].keys():
    #         pts = np.array(initialised[key][k2])
    #         plt3d.plot(pts[:,0],pts[:,1],pts[:,2])
    plt.show()







def fit_subject(subdata,lines, mesh_r,mesh_l, sigmoid, pdf):
    """fits subject meshes"""
    ###right
    se_r = subdata['right_el_names']
    sd_r = subdata['right']
    displaced_els = displace_points_along_electrodes(el_name=se_r,distances=subdata['right_distances'],lines=lines['right'])
    plot_electrode_as_pts(displaced_els,sd_r,mesh_r,mask=subdata['right_masks'])
    res_mesh = fit_side(displaced_els,sd_r,mesh_r,sigmoid,pdf)
    plot_electrode_as_pts(displaced_els, sd_r, res_mesh,mask=subdata['right_masks'])
    ###left
    se_l = subdata['left_el_names']
    sd_l = subdata['left']
    displaced_els = displace_points_along_electrodes(el_name=se_l,
                                                     distances=subdata['left_distances'],lines=lines['left'])

    res_mesh2 = fit_side(displaced_els,sd_l,mesh_l,sigmoid,pdf)

    return [res_mesh,res_mesh2]






    pass

def fit_subjects(comb_data,entry_target,el_names,sigmoid,pdf):

    cnt = 0
    for sub_i in range(len(safe_subjects)):
        sub = safe_subjects[sub_i]
        st_sub = None
        if sub < 100:
            st_sub = "0" + str(sub)
        else:
            st_sub = str(sub)
        fl = open("/home/varga/processing_data/new_data_sorted/sub-P" + st_sub + "/transformACPC","rb")
        to_acpc = pickle.load(fl)


        initialised = generate_lines(toacpc=to_acpc,right=entry_target["right"][cnt]
                                     ,left=entry_target["left"][cnt],
                                     el_names_r=el_names["right"][cnt],el_names_l=el_names["left"][cnt])

        mesh1 = ExtPy.cMesh("/home/varga/processing_data/new_data_sorted/sub-P" + st_sub + "/" + "3_1T1.obj")
        mesh2 = ExtPy.cMesh("/home/varga/processing_data/new_data_sorted/sub-P" + st_sub + "/" + "4_1T1.obj")
        a = mesh1.distance_to_point(1,1,1)
        plot_electrodes(initialised,[mesh2,mesh1])


        t_dict = {}
        for key in comb_data.keys():
            t_dict[key] = comb_data[key][sub_i]

        meshes = fit_subject(subdata=t_dict,lines=initialised,mesh_r = mesh2,mesh_l=mesh1,sigmoid=sigmoid,pdf=pdfs)

        plot_electrodes(initialised, meshes)

        cnt+=1



#def cut_lines()







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # parse_parameters

    preprocessed_data = read_all_subjects()

    comb_data = combine_into_one_array(preprocessed_data)
    pdfs = train_lognorms(comb_data)
    sigmoid = train_sigmoid(preprocessed_data)
    subj_names, ent_targ = read_edf_entry_target("/home/varga/processing_data/participants-ED2.xlsx.ods")
    lenghts = parse_lengths(subj_names, ent_targ)


    fit_subjects(preprocessed_data,lenghts,el_names={"right": preprocessed_data["right_el_names"],"left":preprocessed_data["left_el_names"]}
                 ,sigmoid=sigmoid,pdf=pdfs)



    # a.get_anat_landmarks()
    # #a.rescale_signals()
    #
    # runner = proc.Processor()
    # runner.set_data(a)
    # runner.set_processes([
    #                       ad.covariance_method,
    #                       dat.normalise_mean_std,
    #                       fe.nrms_calculation])
    # a =runner.run()
    # dat = a.get_data()
    # for i in range(a.extracted_features.shape[0]):
    #     plt.plot(a.distances,a.extracted_features[i])
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
