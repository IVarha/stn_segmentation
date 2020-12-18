# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
import sys

import fsl.data.image as fim
import fsl.transform.flirt as fl
import numpy as np

import bayessian_appearance.mesh as mesh



def read_train_subjects(file_name):
    fm = open(file=file_name,mode='rt')
    res = []
    for sub in fm:
        res.append(str(sub.strip('\n')))
    fm.close()
    return res


def read_label_desc(file_name):
    fm = open(file=file_name,mode='rt')
    res = []
    for sub in fm:
        a = sub.split(',')
        res.append(a[0])
    fm.close()
    return res

def read_obj_mesh(file_name):
    pass


#generates point array
def generate_mask(x_r,y_r,z_r,dt):

    x_min = x_r[0]
    x_max = x_r[1]

    x_arr = []
    i = 0
    while True:
        x_arr.append(x_min + i * dt)
        if (x_min + i * dt) >= x_max:
            break
        i += 1
    y_min = y_r[0]
    y_max = y_r[1]
    y_arr = []
    i = 0
    while True:
        y_arr.append(y_min + i * dt)
        if (y_min + i * dt) >= y_max:
            break
        i += 1

    z_min = z_r[0]
    z_max = z_r[1]
    z_arr = []
    i = 0
    while True:
        z_arr.append(z_min + i * dt)
        if (z_min + i * dt) >= z_max:
            break
        i += 1

    res_arr = np.zeros((len(x_arr),len(y_arr),len(z_arr),3))
    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            for k in  range(len(z_arr)):
                res_arr[i,j,k,0] = x_arr[i]
                res_arr[i, j, k, 1] = y_arr[j]
                res_arr[i, j, k, 2] = z_arr[k]
    return res_arr

def create_subj_mask(point_coords, tr_subjects,mesh_name):
    dt = 0.5

    x_r = [point_coords[0],point_coords[1]]
    y_r = [point_coords[2],point_coords[3]]
    z_r = [point_coords[4],point_coords[5]]

    point_array = generate_mask(x_r,y_r,z_r,dt)

    #generate arrays
    #accumulative array
    summ_coord = np.zeros([point_array.shape[0],point_array.shape[1],point_array.shape[2]])

    #bool mask
    bool_masks = []
    for i in range(len(tr_subjects)):
        surf  =  mesh.Mesh(tr_subjects[i] + os.sep + mesh_name + "_1.obj")

        im_ref = fim.Image(tr_subjects[i] + os.sep + "t1_acpc_extracted.nii.gz")
        im_mni = fim.Image(tr_subjects[i] + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")

        forward_transf_fsl = fl.readFlirt(tr_subjects[i] + os.sep + "combined_affine_t1.mat")

        to_mni = fl.fromFlirt(forward_transf_fsl,src=im_ref,ref=im_mni,from_="world",to="world")

        surf.apply_transform(to_mni)
        val_mat = surf.points_is_inside(point_array)
        summ_coord += val_mat
        bool_masks.append( val_mat.astype(np.bool))

    arr_overlapres =[]
    for i in range(len(tr_subjects)):
        point = summ_coord[bool_masks[i]==True].sum()
        arr_overlapres.append(point)
    arr_overlapres = np.array(arr_overlapres)
    arr_overlapres = arr_overlapres / arr_overlapres.max()
    return arr_overlapres














def main_proc(train, label_names, workdir):
    tr_subjects = read_train_subjects(train)
    labels = read_label_desc(label_names)

    #get bounding_borders in pdm s

    meshes = []
    for sub_i in range(len(tr_subjects)):
        #read im
        im_ref = fim.Image(tr_subjects[sub_i] + os.sep + "t1_acpc_extracted.nii.gz")
        im_mni = fim.Image(tr_subjects[sub_i] + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")

        forward_transf_fsl = fl.readFlirt(tr_subjects[sub_i] + os.sep + "combined_affine_t1.mat")

        to_mni = fl.fromFlirt(forward_transf_fsl,src=im_ref,ref=im_mni,from_="world",to="world")
        #get all maxs
        sub_mi_ma = []
        for lab_i in range(len(labels)):
            surf = mesh.Mesh(filename=    tr_subjects[sub_i] + os.sep + labels[lab_i] + "_1.obj" ) #read mesh
            surf.apply_transform(mat=to_mni)
            sub_mi_ma.append(surf.get_min_max())
        meshes.append(sub_mi_ma)

    meshes = np.array(meshes)
    m_mx = []


    # find overlaps
    overlap_stat = []
    for lab_i in range(len(labels)):
        xmin = min(meshes[:,lab_i,0])
        xmax = max(meshes[:,lab_i,1])

        ymin = min(meshes[:,lab_i,2])
        ymax = max(meshes[:,lab_i,3])

        zmin = min(meshes[:,lab_i,4])
        zmax = max(meshes[:,lab_i,5])


        ov = create_subj_mask([xmin,xmax,ymin,ymax,zmin,zmax],tr_subjects,labels[lab_i])
        overlap_stat.append(ov)


    overlap_stat = np.array(overlap_stat)
    os.remove(workdir + os.sep + "overlaped.mat")
    np.savetxt(workdir + os.sep + "overlaped.mat",X=overlap_stat)
    mesh_min_max= np.array(m_mx)












# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi PyCharm')
    train_subjects_file = sys.argv[1]
    labels_desc_file = sys.argv[2]
    outp = sys.argv[3]

    main_proc(train_subjects_file,labels_desc_file,outp)


    # Print some basic information about the layout


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
