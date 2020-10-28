# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from bids import BIDSLayout
from bids.tests import get_test_data_path
import os
import sys
import glob
import nrrd
import nibabel as nib
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    nrrd_file = sys.argv[1]
    nifty_inp = sys.argv[2]
    outp = sys.argv[3]


    nrrd_in = nrrd.read(nrrd_file)

    nif = nib.load(nifty_inp)
    nif2 = nib.Nifti1Image(nrrd_in[0], np.eye(4))
    nib.save(nif2,outp)
    # Initialize the layout
    print(1222)

    # Print some basic information about the layout


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
