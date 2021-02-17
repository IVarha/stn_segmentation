# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys

import bayessian_appearance.utils as util
from datetime import datetime

import bayessian_appearance.point_distribution as pd
import os
import bayessian_appearance.fitting as fitt

import bayessian_appearance.mesh as mesh

import  numpy as np




def main_proc( test_subjects, label_names):
    test_subjs = util.read_subjects(test_subjects)

    labels = util.read_label_desc(label_names)

    #functions
    dice_f = lambda x, y: 2 * (x & y).sum() / (x.sum() + y.sum())
    jac_f = lambda x, y:  (x & y).sum() / (x | y).sum()

    for sub_i in range(len(test_subjs)):


        res = []
        for lab1 in range(len(labels)):
            try:
                tmp = []
                mesh_labeled = mesh.Mesh(test_subjs[sub_i] + os.sep+  labels[lab1] + "_pca.obj")

                mesh_fitted = mesh.Mesh(test_subjs[sub_i] + os.sep + labels[lab1] + "_fitted.obj")
                tmp.append( mesh.Mesh.meshes_overlap(mesh_labeled,mesh_fitted,dice_f))
                tmp.append(mesh.Mesh.meshes_overlap(mesh_labeled, mesh_fitted, jac_f))

                res.append(tmp)
            except:
                pass

        try:
            os.remove(test_subjs[sub_i] + os.sep + "overlap.csv")
        except:
            pass
        np.savetxt(fname=test_subjs[sub_i] + os.sep + "overlap.csv" ,X=np.array(res),delimiter=',' )



    #meshes = np.array(meshes)
    m_mx = []



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_subjects_file = sys.argv[1]
    labels_desc_file = sys.argv[2]
    modalities_name = sys.argv[3]
    test_subjects_file = sys.argv[4]
    a = util.read_segmentation_config(modalities_name)
    main_proc(test_subjects=test_subjects_file,
              label_names=labels_desc_file)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
