# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from bids import BIDSLayout
from bids.tests import get_test_data_path
import os
import nibabel.nifti1

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    data_path = os.path.join(get_test_data_path(), '7t_trt')

    # Initialize the layout
    layout = BIDSLayout(data_path)
    print(1222)

    # Print some basic information about the layout


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
