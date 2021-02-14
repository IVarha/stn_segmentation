# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys

import bayessian_appearance.utils as util
from datetime import datetime

import bayessian_appearance.point_distribution as pd
import os
import bayessian_appearance.fitting as fitt


def main_proc(train_subjects, test_subjects, label_names, config_name, modalities, workdir):
    tr_subjects = util.read_subjects(train_subjects)
    test_subjs = util.read_subjects(test_subjects)

    labels = util.read_label_desc(label_names)

    mod = util.read_modalities_config(modalities)
    seg_cnf = util.read_segmentation_config(modalities)
    cnf = util.read_config_ini(config_name)
    import ExtPy
    tri = [[0, 0, 0],
           [0, 0, 2],
           [0, 2, 0],

           [1, 0, 0],
           [-1, 0, 0],
           [-0.5, 0.5, 0.5]]
    a = ExtPy.is_triangle_intersected(tri)

    meshes = []
    fitter = fitt.Fitter(tr_subjects, test_subj=test_subjs)
    fitter.read_pdm(workdir + os.sep + "pdm.pysave")
    fitter.set_modalities(mod)
    fitter.set_overlaped(workdir + os.sep + "overlaped.mat")
    print(datetime.now())

    fitter.fit_single()

    fitter.fit_multiple()

    print(datetime.now())

    # meshes = np.array(meshes)
    m_mx = []


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_subjects_file = sys.argv[1]
    labels_desc_file = sys.argv[2]
    conf_file = sys.argv[3]
    outp = sys.argv[4]
    modalities_name = sys.argv[5]
    test_subjects_file = sys.argv[6]
    a = util.read_segmentation_config(modalities_name)
    main_proc(train_subjects=train_subjects_file, test_subjects=test_subjects_file,
              label_names=labels_desc_file, config_name=conf_file,
              modalities=modalities_name, workdir=outp)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
