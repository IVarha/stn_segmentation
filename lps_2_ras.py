# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import SimpleITK as sitk
import sys

import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def lps_2_ras_coord(lps_coord):
    return  [lps_coord[0]*(-1),lps_coord[1]*(-1),lps_coord[2]]


def read_tfm(file_name):
    with open(file_name,'rt') as f:
        f.readline()
        f.readline()
        f.readline()
        ln = f.readline()
        stg2 = ln.split(' ')


        matx = []
        for i in range(1,len(stg2)):
            matx.append(float(stg2[i]))
        matx = matx
        matx = np.array(matx)
        matx = matx.reshape((4,3)).transpose().tolist() + [[0,0,0,1]]
        matx = np.array(matx)
        return matx
        print(1)







def lps_to_coord_file(filename,out_file,tfm):
    lines = [ ]
    with open(filename,'rt') as f:


        h1 = f.readline()
        f.readline()
        h2 = "# CoordinateSystem = RAS\n"
        h3 = f.readline()
        lines = f.readlines()
        transf = sitk.ReadTransform(tfm)
        for i in range(len(lines)):

            ln = lines[i]

            sng = ln.split(',')

            coord = [float(sng[1]),float(sng[2]),float(sng[3])]


            coord2 = lps_2_ras_coord(coord)
            coord_transf = np.dot(transf, np.array(coord + [1]))
            sng[1] = str(coord2[0])
            sng[2] = str(coord2[1])
            sng[3] = str(coord2[2])
            res_str = ','.join(sng)
            lines[i] = res_str
        lines = [h1,h2,h3] + lines
    with open(out_file,'wt') as f2:

        f2.writelines(lines)








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    file_in = sys.argv[1]
    outp = sys.argv[2]
    tfm=sys.argv[3]
    lps_to_coord_file(file_in,out_file=outp,tfm=tfm)
    # Initialize the layout
    print(1222)

    # Print some basic information about the layout


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
