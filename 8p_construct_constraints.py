# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys

import bayessian_appearance.utils as util

import bayessian_appearance.point_distribution as pd
import os

def main_proc(train, label_names, config_name,modalities, workdir):
    tr_subjects = util.read_subjects(train)
    labels = util.read_label_desc(label_names)

    mod = util.read_modalities_config(modalities_name)

    seg_cnf = util.read_segmentation_config(modalities_name)
    cnf = util.read_config_ini(config_name)

    meshes = []
    pdm = pd.PointDistribution(train_subjects=tr_subjects,labels=labels,segmentation_conf=seg_cnf)

    pdm.save_pdm(workdir + os.sep + "pdm.pysave")
        # read im




    #meshes = np.array(meshes)
    m_mx = []



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_subjects_file = sys.argv[1]
    labels_desc_file = sys.argv[2]
    conf_file = sys.argv[3]
    outp = sys.argv[4]
    modalities_name = sys.argv[5]
    a = util.read_segmentation_config(modalities_name)
    main_proc(train_subjects_file, labels_desc_file, conf_file,modalities=modalities_name, workdir=outp)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
