# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys

import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def lps_2_ras_coord(lps_coord):
    return  [lps_coord[0]*(-1),lps_coord[1]*(-1),lps_coord[2]]

"""Returns transform LPS"""
def read_tfm_LPS(file_name):

    matr = np.zeros((4,4))
    with open(file_name,'rt') as f:
        f.readline()
        f.readline()
        f.readline()
        ln = f.readline()
        stg2 = ln.split(' ')


        matx = []
        matr[0,0] = float(stg2[1])
        matr[0, 1] = float(stg2[2])
        matr[0, 2] = float(stg2[3])
        matr[0, 3] = float(stg2[10])

        matr[1, 0] = float(stg2[4])
        matr[1, 1] = float(stg2[5])
        matr[1, 2] = float(stg2[6])
        matr[1, 3] = float(stg2[11])

        matr[2, 0] = float(stg2[7])
        matr[2, 1] = float(stg2[8])
        matr[2, 2] = float(stg2[9])
        matr[2, 3] = float(stg2[12])
        matr[3, 0] = 0
        matr[3, 1] = 0
        matr[3, 2] = 0
        matr[3, 3] = 1



        return matr








def lps_to_coord_file(filename,out_file,tfm):
    lines = [ ]
    with open(filename,'rt') as f:


        h1 = f.readline()
        f.readline()
        h2 = "# CoordinateSystem = RAS\n"
        h3 = f.readline()
        lines = f.readlines()
        transf = read_tfm_LPS(tfm)
        # 4 lines only
        t2 = ['','','','']
        t3 = []
        for i in range(8):

            ln = lines[i]

            sng = ln.split(',')

            coord = [float(sng[1]),float(sng[2]),float(sng[3])]



            coord_transf = np.dot(np.linalg.inv(transf), np.array(coord + [1]))

            coord2 = lps_2_ras_coord(coord_transf)
            sng[1] = str(coord2[0])
            sng[2] = str(coord2[1])
            sng[3] = str(coord2[2])

            res_str = ','.join(sng)
            t3.append(res_str)
            #print(res_str)
            lines[i] = res_str
        # t2[0] = t3[1]
        # t2[1] = t3[0]
        # t2[2] = t3[3]
        # t2[3] = t3[2]
        # print(','.join(t2))
        lines = [h1,h2,h3] + lines
    with open(out_file,'wt') as f2:
        for i in lines:
            f2.write(i)








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    file_in = sys.argv[1]
    outp = sys.argv[2]
    tfm=sys.argv[3]
    lps_to_coord_file(file_in,out_file=outp,tfm=tfm)
    # Initialize the layout
    #print(1222)

    # Print some basic information about the layout


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
