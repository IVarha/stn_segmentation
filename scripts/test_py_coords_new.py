# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import lps_2_ras as l2r
import bayessian_appearance.utils as uti


import h5py
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# safe_subjects = [60,62,63,66,69,70,71,73,74,75,77,78,79,81,82,83,84,85,86,87,90,93,95,97,98,99,100,102,104,105,106,107,110,111,112,113,114,115,116,118,120,125,126,129,132,133]
safe_subjects = [60, 62, 63, 66, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 90, 93, 95, 97, 98, 99,
                 102, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 116, 118, 120, 125, 126, 129, 132, 133]


# safe_subjects = [60]

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

def plot_3d_lines(triangle, triangle2):
    triangle1 = triangle
    tri_test = [ type(x) == type([]) for x in triangle1]
    if max(tri_test)== True:
        triangle1 = [np.array(x) for x in triangle]
    plt3d = plt.figure().gca(projection='3d')

    ln1 = np.array([triangle1[0].tolist(), triangle1[1].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2],c='r')
    ln1 = np.array([triangle1[0].tolist(), triangle1[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2],c='b')
    ln1 = np.array([triangle1[1].tolist(), triangle1[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2],c='g')

    ln1 = np.array([triangle2[0].tolist(),triangle2[1].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2],c='r')
    ln1 = np.array([triangle2[0].tolist(),triangle2[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2],c='b')
    ln1 = np.array([triangle2[1].tolist(),triangle2[2].tolist()])
    plt3d.plot(ln1[:,0],ln1[:,1],ln1[:,2],c='g')

    plt.show()

def generate_transform_coords_to_t1(center,left,top):

    c_orig = np.array([100,100,100])
    c_left = np.array([101,100,100])
    c_top = np.array([100, 101, 100])

    center1 = center
    left1 = left
    top1 = top
    c_vec = c_orig - center
    to_c_orig = uti.translate_p(c_vec)
    from_corig = np.linalg.inv(to_c_orig)

    res_mat = to_c_orig
    center = uti.apply_transf_2_pts([center.tolist()],to_c_orig)[0]
    left = uti.apply_transf_2_pts([left.tolist()],to_c_orig)[0]
    top = uti.apply_transf_2_pts([top.tolist()], to_c_orig)[0]

    c1 = np.array(center)
    c2 = np.array(left)
    c3 = np.array(top)

    a1 = c1 - c2
    a1 = a1/np.linalg.norm(a1)
    c2 = c1 - a1

    a2 = c3 - c2

    a2 = a2 / np.linalg.norm(a2)
    c3 = c2 + a2
    a1 = a1[0:3]
    a2 = a2[0:3]
    a_cross = np.cross(a1, a2)
    a_cross = a_cross/np.linalg.norm(a_cross)
    ########90degrees fix
    p1 = uti.translate_p(-c1)
    pr = uti.translate_p(c1)
    rot_90 = uti.rotate_axis(a_cross,-np.pi/2)
    mult_mat = np.linalg.multi_dot([pr,rot_90,p1])

    c3 = np.array(uti.apply_transf_2_pts([c2.tolist()],mult_mat)[0])


    z1 = np.array(c_orig)
    z2 = np.array(c_left)
    z3 = np.array(c_top)
    plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    plot_3d_plane([c1,c2,c3],[z1,z2,z3])

    mni_a1 = z1 - z2
    mni_a2 = z3 - z2
    mni_across = np.cross(mni_a1, mni_a2)
    mni_across = mni_across/np.linalg.norm(mni_across)

    d1 = -np.dot(a_cross, c1)
    d2 = -np.dot(mni_across, z1)
    res_intr = uti.plane_intersect(list(a_cross) + [d1], list(mni_across) + [d2])
    p = res_intr[0]
    N = (res_intr[1] - res_intr[0]) / np.linalg.norm(res_intr[1] - res_intr[0])

    ###angle between
    #np.cross(c1 - c2,)

    #angle between two planes (MNI coords and MNI origin PLANE)
    alpha = np.arccos(np.dot(mni_across, a_cross) / (np.linalg.norm(mni_across) * np.linalg.norm(a_cross)))
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)

    print(alpha)

    rotA = uti.rotate_axis(N, alpha)

    c1,c2,c3 = uti.apply_transf_2_pts([c1.tolist(),c2.tolist(),c3.tolist()],rotA)
    res_mat = np.dot(rotA,res_mat)
    plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    plot_3d_plane([c1, c2, c3], [z1, z2, z3])
    # match_pts = uti.translate_p(np.array(c_orig)-np.array(c1))
    # c1,c2,c3 = uti.apply_transf_2_pts([c1,c2,c3],match_pts)
    # plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    # plot_3d_plane([c1, c2, c3], [z1, z2, z3])

    alpha = np.arccos(np.dot(np.array(c1)-np.array(c2), np.array(z1)-np.array(z2))
                      / (np.linalg.norm(np.array(c1)-np.array(c2)) * np.linalg.norm(np.array(z1)-np.array(z2))))
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)

    rotB = uti.rotate_axis(mni_across, alpha)
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)

    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], rotB)
    res_mat = np.dot(rotB, res_mat)
    plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    plot_3d_plane([c1, c2, c3], [z1, z2, z3])
    match_pts = uti.translate_p(np.array(c_orig) - np.array(c1))
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], match_pts)
    res_mat = np.dot(match_pts, res_mat)
    # plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    # plot_3d_plane([c1, c2, c3], [z1, z2, z3])

    mir_tr = uti.mirror_point(np.array([99.5,100,100]),0)
    res_mat = np.dot(mir_tr,res_mat)

    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], mir_tr)
    plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    plot_3d_plane([c1, c2, c3], [z1, z2, z3])
    match_pts = uti.translate_p(np.array(c_orig) - np.array(c1))
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], match_pts)
    plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    # plot_3d_plane([c1, c2, c3], [z1, z2, z3])

    res_mat = np.dot(match_pts, res_mat)
    res_mat = np.dot(uti.mirror_point(np.array([100,100,100]),2),res_mat)

    ##rotate
    r1 = np.array(c1) - np.array(c2)
    r2 = z1 - z2
    rotA = uti.rotation_axis_2vecs(c1,r1,r2,mni_across)
    res_mat = np.dot(rotA,res_mat)
    c1, c2, c3 = uti.apply_transf_2_pts([c1, c2, c3], rotA)
    plot_3d_lines([c1, c2, c3], [z1, z2, z3])
    mir_tr = uti.mirror_point(np.array([99.5, 100, 100]), 0)
    res_mat = np.dot(mir_tr, res_mat)

    # plot_3d_plane([c1, c2, c3], [z1, z2, z3])
    #res_mat = np.linalg.multi_dot([match_pts,rotB, rotA])
    #res_mat = np.linalg.multi_dot([rotB,rotA,to_c_orig])
    res_matr1 = np.linalg.inv(res_mat)


    return res_matr1


def read_framecoords(filename):
    lines = [ ]
    resT = None
    resL = None
    resC = None
    with open(filename,'rt') as f:


        h1 = f.readline()
        f.readline()
        h2 = "# CoordinateSystem = RAS\n"
        h3 = f.readline()
        lines = f.readlines()
        # 4 lines only
        t2 = ['','','','']
        t3 = []
        for i in range(8):

            ln = lines[i]

            sng = ln.split(',')

            coord = [float(sng[1]),float(sng[2]),float(sng[3])]
            if sng[11] == 'midE':
                resT = coord
            if sng[11] == 'midH':
                resL = coord
            if sng[11] == 'frame_center':
                resC = coord
    return [resC,resL,resT]

def read_doctors_coords(ods_file,subject):
    Lx = "LActualFrameX"
    Ly = "LActualFrameY"
    Lz = "LActualFrameZ"
    sub = "subject"
    Rx = "RActualFrameX"
    Ry = "RActualFrameY"
    Rz = "RActualFrameZ"

    from pandas_ods_reader import read_ods
    df = read_ods(ods_file, "surgical")

    subs = df[sub].values.astype(int).tolist()
    ind = subs.index(subject)

    return [[df[Rx].values[ind],df[Ry].values[ind],df[Rz].values[ind]],
            [df[Lx].values[ind],df[Ly].values[ind],df[Lz].values[ind]]]




def write_fcsv(coords,filename):
    print(filename)
    lines = [ '# Markups fiducial file version = 4.11\n',
              '# CoordinateSystem = RAS\n',
              '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n',
              ]

    coords[0] = [str(x) for x in coords[0]]
    coords[1] = [str(x) for x in coords[1]]
    s1 = ','.join(coords[0])
    ln1 = "1,"+s1 + ",0,0,0,1,1,1,1,Rtarg_surg,,\n"
    s2 = ','.join(coords[1])
    ln2 = "1,"+s2 + ",0,0,0,1,1,1,1,Ltarg_surg,,\n"
    lines.append(ln1)
    lines.append(ln2)
    with open(filename,'wt') as f2:
        for i in lines:
            f2.write(i)


def lps_2_ras_coord(lps_coord):
    return  [lps_coord[0]*(-1),lps_coord[1]*(-1),lps_coord[2]]

def transf_lps_2_ras(transform):
    transform[0, 2] = -1 * transform[0, 2]
    transform[0, 3] = -1 * transform[0, 3]
    transform[1, 2] = -1 * transform[1, 2]
    transform[1, 3] = -1 * transform[1, 3]
    transform[2, 0] = -1 * transform[2, 0]
    transform[2, 1] = -1 * transform[2, 1]
    return transform

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    file_name = sys.argv[1]
    ods_fn = sys.argv[2]
    out_fn = sys.argv[3]
    surg_transform_f = sys.argv[4]
    to_t1_f = sys.argv[5]

    to_t1_lps = l2r.read_tfm_LPS(to_t1_f)
    ##lps_transf_2_ras
    to_t1_lps = transf_lps_2_ras(to_t1_lps)

    ##h5 parse
    surg_transf_1 = h5py.File(surg_transform_f)
    vals  = [ surg_transf_1[x] for x in surg_transf_1.keys()]
    vals = [vals[4]['0'][x] for x in vals[4]['0'].keys()]

    t1 = np.zeros((4,4))
    t1[0,0] = vals[1][0]
    t1[0, 1] = vals[1][1]
    t1[0, 2] = vals[1][2]
    t1[0, 3] = vals[1][9]
    t1[1, 0] = vals[1][3]
    t1[1, 1] = vals[1][4]
    t1[1, 2] = vals[1][5]
    t1[1, 3] = vals[1][10]
    t1[2, 0] = vals[1][6]
    t1[2, 1] = vals[1][7]
    t1[2, 2] = vals[1][8]
    t1[2, 3] = vals[1][11]
    t1[3, 0] = 0
    t1[3, 1] = 0
    t1[3, 2] = 0
    t1[3, 3] = 1



    fc,left,top = read_framecoords(file_name)
    #fc = lps_2_ras_coord(fc)
    frameToFC = np.array([
        [1, 0, 0, fc[0]],
        [0, 1, 0, fc[1]],
        [0, 0, 1, fc[2]],
        [0, 0, 0, 1]
    ])
    #frameToFC = uti.translate_p(-np.array(fc))
    frameCoordOrigin = np.array([
        [-1, 0, 0, 100],
        [0, 1, 0, -100],
        [0, 0, -1, 100],
        [0, 0, 0, 1]
    ])
    # cen = np.array([100,100,100])
    # frameCoordOrigin = uti.translate_p(-cen)
    # to_ras = np.array([
    #     [-1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])
    to_ras = np.eye(4)
    transf_res = np.linalg.multi_dot([
                                    #to_ras,
                                    to_t1_lps,
                                    np.linalg.inv(t1),
                                    frameToFC,
                                    frameCoordOrigin])


    # fc = np.array([-1.5528718082009654, 48.59929770528181, 6.085721070353887])
    # left = np.array([-96.21642270509193, 57.873290233812185, 6.571655602652973])
    # top = np.array([9.776208841309534, 163.0478945820811, 2.755798118783989])
    #res  = generate_transform_coords_to_t1(np.array(fc),np.array(left),np.array(top))
    coords = read_doctors_coords(ods_fn,int(file_name.split('/')[-2].split('-')[-1][1:]))
    coords = uti.apply_transf_2_pts(coords,transf_res)
    coords = [lps_2_ras_coord(x) for x in coords]
    write_fcsv(coords,out_fn)


    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
