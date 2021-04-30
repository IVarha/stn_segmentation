
"""GENERATE NEW IMAGE FROM STATISTICS"""

import sys
import nibabel as nib
import numpy as np
def main_proc(inp, outp):

    inp_file = nib.load(inp)

    inp_im = inp_file.get_fdata()



    X_cenre = np.dot(np.linalg.inv(inp_file.affine),np.array([0,0,0,1]))[0]

    arr_X = []
    for i in range(inp_im.shape[0]):
        if inp_im[i,:,:].max() == 1:
            arr_X.append(i)
    min_x = min(arr_X)
    max_x = max(arr_X)

    arr_X = []
    for i in range(inp_im.shape[1]):
        if inp_im[:,i,:].max() == 1:
            arr_X.append(i)
    min_y = min(arr_X)
    max_y = max(arr_X)

    arr_X = []
    for i in range(inp_im.shape[2]):
        if inp_im[:,:,i].max() == 1:
            arr_X.append(i)
    min_z = min(arr_X)
    max_z = max(arr_X)

    if abs(X_cenre - min_x) >= abs(max_x - X_cenre):
        h = int(abs(X_cenre - min_x))
        max_x = int(X_cenre + h)

    else:
        h = int(abs(max_x - X_cenre))
        min_x = int(X_cenre - h)

    inp_im[min_x:max_x,min_y:max_y,min_z:max_z] = 1

    img = nib.Nifti1Image(inp_im.astype(np.int), inp_file.affine)
    nib.save(img, outp)


    # read im

    # meshes = np.array(meshes)
    m_mx = []


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_file = sys.argv[1]
    outputfile = sys.argv[2]

    main_proc(input_file,outputfile)
