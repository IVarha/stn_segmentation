

import bayessian_appearance.point_distribution as pd
import bayessian_appearance.mesh as mesh
import bayessian_appearance.utils as utils

import numpy as np
import bayessian_appearance.vt_image as vtimage
import fsl.data.image as fim
import fsl.transform.flirt as fl
import os
import nibabel as nib


import bayessian_appearance.settings as gl_set

import scipy.optimize as opt
#for single mesh
class FunctionHandler:

    _image = None
    _from_mni_to_vox = None
    _vox_to_world = None
    #label of single mesh
    _label = None
    _mesh = None
    _num_of_points=None


    _constraints = None
    _kdes = None

    def __init__(self):
        pass

    def set_image(self, filename):

        vt = vtimage.Image(filename)
        vt.setup_bspline(3)
        self._image = vt

        a = nib.load(filename)
        self._vox_to_world = a.affine


    #TODO rewrite to make it universal
    def set_subject(self,subject_dir):

        mni_im = fim.Image(subject_dir + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz" )

        native_im = fim.Image(subject_dir + os.sep + "t1_acpc_extracted.nii.gz" )

        from_mni_mat = fl.readFlirt(subject_dir + os.sep + "combined_affine_reverse.mat")


        mni_w = fl.fromFlirt(from_mni_mat,mni_im,native_im,"world","world")
        a = np.linalg.inv(self._vox_to_world)
        self._from_mni_to_vox = np.dot(a,mni_w)


    def __call__(self, *args, **kwargs):
        coords = np.array(args[0])
        coords = coords.reshape((self._num_of_points,3))


        self._mesh.modify_points(coords)

        normals = self._mesh.generate_normals(gl_set.settings.norm_length,gl_set.settings.discretisation)
        normals = utils.apply_transf_2_norms(normals,self._from_mni_to_vox)

        norm_intens = np.array(self._image.interpolate_normals(normals))
        res = -1

        for i in range(self._num_of_points):
            p1 = self._kdes[0][i].pdf(coords[i,:])
            p2 = self._kdes[1][i].pdf(norm_intens[i,:])

            print(1)


















class Fitter:

    _pdm = None

    _train_subj = None
    _test_subj = None
    _modal = None
    _mat_over = None
    _best_meshes_mni = None



    def __init__(self,train_subj,test_subj,pdm=None):
        self._train_subj = train_subj
        self._test_subj = test_subj
        if pdm != None:
            if isinstance(pdm,str):
                self._pdm = pd.PointDistribution.read_pdm(file_name=pdm)

            else:
                self._pdm = pdm

    def read_pdm(self, filename ):
        self._pdm = pd.PointDistribution.read_pdm(file_name=filename)


    def set_modalities(self, modalities):
        self._modal = modalities


    def set_overlaped(self,file_name):
        mat = np.loadtxt(file_name)
        self._mat_over = mat

        #get best meshes of segmented figures
        self._best_meshes_mni = []
        for i in range(len(gl_set.settings.labels_to_segment)):
            label = gl_set.settings.labels_to_segment[i]
            ind = gl_set.settings.all_labels.index(label)

            sub_ind = list(self._mat_over[ind]).index(1)
            mni_im = fim.Image(self._train_subj[sub_ind] + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")

            native_im = fim.Image(self._train_subj[sub_ind] + os.sep + "t1_acpc_extracted.nii.gz")

            to_mni_mat = fl.readFlirt(self._train_subj[sub_ind] + os.sep + "combined_affine_t1.mat")

            mni_w = fl.fromFlirt(to_mni_mat, native_im, mni_im, "world", "world")

            surf = mesh.Mesh(self._train_subj[sub_ind] + os.sep + label + "_1.obj")
            surf.apply_transform(mni_w)
            self._best_meshes_mni.append(surf)





    def fit_single(self):

        for i in range(len(self._test_subj)):

            for lab in range(len(self._best_meshes_mni)):
                fc = FunctionHandler()
                #todo fix set
                fc.set_image(  self._test_subj[i] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz"   )
                fc.set_subject(self._test_subj[i])
                fc._mesh = self._best_meshes_mni[lab]
                fc._kdes = self._pdm.get_kdes()[lab]
                fc._num_of_points= self._best_meshes_mni[lab].gen_num_of_points()

                X0 = fc._mesh.get_unpacked_coords()

                a = fc(X0)



            pass







        pass