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


def train_test_file_create(subjects, ind_of_subject,out_train_file, out_test_file):
    subs = subjects.copy()

    sub = subs[ind_of_subject]

    subs.remove(sub)

    subs_tr = subs

    subs_test = [sub]


    write_subjs_2_file(subs_tr,out_train_file)
    write_subjs_2_file(subs_test,out_test_file)



def main_proc( subjects, train_file_name, test_file_name, script_name, start_sub_name = None):
    subjs = util.read_subjects(subjects)

    mark = False
    for sub_i in range(len(subjs)):
        if start_sub_name is None:

            train_test_file_create(subjs,sub_i,train_file_name,test_file_name)

            run_script( script_name)
        else:
            if subjs[sub_i] == start_sub_name:
                mark = True
                train_test_file_create(subjs, sub_i, train_file_name, test_file_name)
                run_script(script_name)
                continue

            if mark:
                train_test_file_create(subjs, sub_i, train_file_name, test_file_name)
                run_script(script_name)




    #meshes = np.array(meshes)
    m_mx = []



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    subjects = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    script_nm = sys.argv[4]
    start_sub = None
    if len(sys.argv)>5:
        start_sub = sys.argv[5]
    main_proc(subjects=subjects,train_file_name=train_file,test_file_name=test_file,script_name=script_nm,start_sub_name=start_sub)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
