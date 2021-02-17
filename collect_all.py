# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys
import pandas as pand
import bayessian_appearance.utils as util
from datetime import datetime

import bayessian_appearance.point_distribution as pd
import os
import bayessian_appearance.fitting as fitt

import bayessian_appearance.mesh as mesh

import  numpy as np

import subprocess




def run_script(script):
    proces= "source /home/varga/venvs/pystn/bin/activate;"
    subprocess.call(proces + script,shell=True,executable="/bin/bash")


def write_subjs_2_file(subjs, filename):

    try:
        os.remove(filename)
    except:
        pass

    f = open(file=filename,mode='wt')
    for i in range(len(subjs)):
        f.write(subjs[i] + "\n")


def overlap_load(subject):
    arr = None
    try:
        arr = np.loadtxt(subject + os.sep + "overlap.csv",delimiter=',')
    except:
        pass

    return arr




def main_proc( subjects, label_names, workdir ):
    subjs = util.read_subjects(subjects)

    labels = util.read_label_desc(label_names)

    res_arr = []
    for label in labels:
        res_arr.append(None)



    for sub_i in range(len(subjs)):
        a = overlap_load(subjs[sub_i])
        if a is None:
            pass
        else:
            for lab_i in range(len(labels)):

                if res_arr[lab_i] is None:
                    res_arr[lab_i] = []
                    res_arr[lab_i].append([subjs[sub_i]]+ a[lab_i,:].tolist())
                else:
                    res_arr[lab_i].append([subjs[sub_i]]+ a[lab_i, :].tolist())
            pass
    for i in range(len ( res_arr )):
        res_arr[i] = np.array(res_arr[i])
        pand.DataFrame(res_arr[i]).to_csv(workdir+os.sep + labels[i] + "over.csv")
        #np.savetxt(X=res_arr[i],fname=workdir+os.sep + labels[i] + "over.csv",delimiter=',')









    #meshes = np.array(meshes)
    m_mx = []



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    subjects = sys.argv[1]
    label_names = sys.argv[2]
    workdir = sys.argv[3]
    script_nm = sys.argv[4]
    main_proc(subjects=subjects,label_names=label_names,workdir=workdir)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
