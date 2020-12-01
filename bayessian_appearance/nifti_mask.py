import nibabel as nib

import numpy as np


class ni_mask:

    __im_mask = None
    __im_vox_world = None
    __im_world_voxel = None
    def __init__(self,file_name):
        a = nib.load(file_name)
        self.__im_mask = (a.get_fdata()).astype(np.bool)
        self.__im_vox_world = a.affine
        self.__im_world_voxel = np.linalg.inv (a.affine)


    def check_neighbours(self, voxel):
        v1 = [int(voxel[0]),int(voxel[1]),int(voxel[2])]
        if (((v1[0]+1) > (self.__im_mask.shape[0])) or
                ((v1[1]+1) > (self.__im_mask.shape[1])) or
                ((v1[2]+1) > (self.__im_mask.shape[2]))):
            return False
        if ( (v1[0]< 0) or (v1[1] <0) or (v1[2] <0)  ):
            return False
        a = self.__im_mask[v1[0],v1[1],v1[2]]
        a = a and self.__im_mask[v1[0], v1[1], v1[2] + 1]
        a = a and self.__im_mask[v1[0], v1[1]+1, v1[2] ]
        a = a and self.__im_mask[v1[0], v1[1]+1, v1[2] + 1]
        a = a and self.__im_mask[v1[0]+1, v1[1], v1[2]]
        a = a and self.__im_mask[v1[0]+1, v1[1], v1[2] + 1]
        a = a and self.__im_mask[v1[0]+1, v1[1]+1, v1[2]]
        a = a and self.__im_mask[v1[0]+1, v1[1]+1, v1[2] + 1]
        return a

    def check_neighbours_world(self, voxel):
        v1 = np.array(voxel + [1])
        v1 = np.dot(self.__im_world_voxel,v1)
        v1 = list(v1[:3])
        return self.check_neighbours(v1)