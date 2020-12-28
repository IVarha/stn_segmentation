import nibabel as nib


import sys

import numpy as np
import os

import bayessian_appearance.mesh as ms
import bayessian_appearance.utils as util
def read_atlas_list(file_name):
    fm = open(file=file_name,mode='rt')
    res = []
    for sub in fm:
        a = sub.split(',')
        res.append([a[0],a[1][:-1]])
    fm.close()
    return res



def convert_atlast_list_2_meshes(atlases,discret,workdir):

    res = []
    for el in atlases:

        atlas = ms.Mesh(el[1])
        mn_mx = atlas.get_min_max()
        mn_mx[0] = mn_mx[0] - 1
        mn_mx[1] = mn_mx[1] + 1
        mn_mx[2] = mn_mx[2] - 1
        mn_mx[3] = mn_mx[3] + 1
        mn_mx[4] = mn_mx[4] - 1
        mn_mx[5] = mn_mx[5] + 1


        arr = util.generate_mask([mn_mx[0],mn_mx[1]],[mn_mx[2],mn_mx[3]],[mn_mx[4],mn_mx[5]],0.1)



        x_coords=[]
        y_coords = []
        z_coords = []
        for i in range(arr.shape[0]):
            x_coords.append(arr[i,0,0,0])
        for j in range(arr.shape[1]):
            y_coords.append(arr[0,j,0,1])
        for k in range(arr.shape[2]):
            z_coords.append(arr[0, 0, k, 2])
        mask = atlas.points_is_inside(arr)

        sphr = ms.Mesh.generate_sphere(center=atlas.get_centre(),radius=200,discretisation=int(discret))

        sphr.shrink_sphere(mask=mask,coords=[x_coords,y_coords,z_coords],cenre=atlas.get_centre(),tres=0.2)

        sphr.smooth_mesh()
        sphr.save_obj(workdir + "/" + el[0] + "m.obj")




























if __name__ == '__main__':
    atlas_list = sys.argv[1]
    discretisation = sys.argv[2]
    wd = sys.argv[3]

    atlas_l = read_atlas_list(atlas_list)


    convert_atlast_list_2_meshes(atlases=atlas_l,discret=discretisation,workdir=wd)






