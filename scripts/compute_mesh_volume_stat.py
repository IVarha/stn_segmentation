"""GENERATE NEW IMAGE FROM STATISTICS"""
import os
import sys

import ExtPy
import pandas as pd


def main_proc(inp, outp):
    # read im

    # meshes = np.array(meshes)
    m_mx = []


def calculate_statistics_anatomically(workdir, subjects):
    """
    returns segmented volumes and volumes
    """
    mesh_name_1 = "3_1T1.obj"
    mesh_name_2 = "4_1T1.obj"
    mesh_name_3 = "3_fittedT1.obj"
    mesh_name_4 = "4_fittedT1.obj"
    folder = workdir

    # points positions of intersection in T1 space
    l_pts_t1 = []
    r_pts_t1 = []

    # point position of intersectioned subjects
    index = []
    volumes = []
    seg_vols = []
    for i in range(len(subjects)):
        try:
            r_mesh = ExtPy.cMesh(folder + os.sep
                                 + subjects[i] + os.sep + mesh_name_2)
            l_mesh = ExtPy.cMesh(folder + os.sep
                                 + subjects[i] + os.sep + mesh_name_1)

            a = r_mesh.calculate_volume()  # right

            b = l_mesh.calculate_volume()  # left
            index.append(subjects[i])
            volumes.append([a, b])
            r_mesh = ExtPy.cMesh(folder + os.sep
                                 + subjects[i] + os.sep + mesh_name_3)
            l_mesh = ExtPy.cMesh(folder + os.sep
                                 + subjects[i] + os.sep + mesh_name_4)
            a = r_mesh.calculate_volume()  # right

            b = l_mesh.calculate_volume()  # left
            seg_vols.append([a, b])
        except:
            pass

    return pd.DataFrame(data=seg_vols, index=index, columns=["right", "left"]), pd.DataFrame(data=volumes, index=index,
                                                                                             columns=["right", "left"])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_file = sys.argv[1]  # input fold

    subjs = []
    for sub in os.listdir(input_file):
        if sub.startswith("sub-P"):
            subjs.append(sub)

    #######folder####################
    # mesh_name_1 = "3_1T1.obj"
    # mesh_name_2 = "4_1T1.obj"
    #
    mesh_name_1 = "3_fittedT1.obj"
    mesh_name_2 = "4_fittedT1.obj"

    df_seg, df1 = calculate_statistics_anatomically(input_file, subjs)
    df1 = df1[df1.any(axis=1)]

    df_seg = df_seg[df_seg.any(axis=1)]

    out_file = sys.argv[2]
    try:
        os.remove(out_file)
        os.remove(out_file + "seg")
    except:
        pass
    df1.to_csv(out_file)
    df_seg.to_csv(out_file + "seg")
