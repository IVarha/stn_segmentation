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

    subprocess.run(['sh',script])


def write_subjs_2_file(subjs, filename):

    try:
        os.remove(filename)
    except:
        pass

    f = open(file=filename,mode='wt')
    for i in range(len(subjs)):
        f.write(subjs[i] + "\n")


def train_test_file_create(subjects, ind_of_subject,out_train_file, out_test_file):
    subs = subjects

    sub = subs[ind_of_subject]

    subs.remove(sub)

    subs_tr = subs

    subs_test = [sub]


    write_subjs_2_file(subs_tr,out_train_file)
    write_subjs_2_file(subs_test,out_test_file)



def main_proc( subjects, train_file_name, test_file_name, script_name):
    subjs = util.read_subjects(subjects)

    for sub_i in range(len(subjs)):
        train_test_file_create(subjs,sub_i,train_file_name,test_file_name)

        run_script( script_name)
    #meshes = np.array(meshes)
    m_mx = []



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    subjects = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    script_nm = sys.argv[4]
    main_proc(subjects=subjects,train_file_name=train_file,test_file_name=test_file,script_name=script_nm)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/