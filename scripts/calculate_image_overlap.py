# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys

import bayessian_appearance.utils as util

import os

import nibabel as nib
import  numpy as np




def main_proc( subjects, orig_label,relabel,workdir):
    test_subjs = util.read_subjects(subjects)




    #functions
    dice_f = lambda x, y: 2 * (x & y).sum() / (x.sum() + y.sum())
    jac_f = lambda x, y:  (x & y).sum() / (x | y).sum()
    st = None
    res = None

    for sub_i in range(len(test_subjs)):
        wd = test_subjs[sub_i]
        orig_label1_file = nib.load(wd + os.sep + orig_label)
        orig_relabelled = nib.load(wd + os.sep + relabel)
        if st is None:
            st = np.unique(orig_relabelled.get_fdata()).tolist()
            st = np.delete(st, 0)
            res = []
            for l in st:
                res.append([])
        for label_i in range(len(st)):
            orig_label_lab = orig_label1_file.get_fdata() == st[label_i]
            orig_relab_lab = orig_relabelled.get_fdata() == st[label_i]

            res[label_i].append( dice_f(orig_label_lab,orig_relab_lab))







    for label_i in range(len(st)):



        try:
            os.remove(workdir + os.sep +str(st[label_i])+ "overlap.csv")
        except:
            pass
        np.savetxt(fname=workdir + os.sep +str(st[label_i]) + "overlap.csv" ,X=np.array(res),delimiter=',' )



    #meshes = np.array(meshes)
    m_mx = []



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_subjects_file = sys.argv[1]
    workdir = sys.argv[2]
    orig_label = sys.argv[3]
    relabel = sys.argv[4]

    main_proc(subjects=train_subjects_file,orig_label=orig_label
              ,workdir=workdir,relabel=relabel)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
