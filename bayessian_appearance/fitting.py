

import bayessian_appearance.point_distribution as pd
import bayessian_appearance.mesh as mesh
import bayessian_appearance.utils as utils

import numpy as np
import bayessian_appearance.vt_image as vtimage
import fsl.data.image as fim
import fsl.transform.flirt as fl
import os
import nibabel as nib

from datetime import datetime

import sys

sys.path.insert(0, "/tmp/tmp.9HaHyiykJ1/cmake-build-debug-remote-host")
import ExtPy


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
    _cmesh = None
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


        coords = self._kdes.vector_2_points(coords)
        coords2 = coords.reshape((self._num_of_points,3))


        self._mesh.modify_points(coords2)

        normals = self._mesh.generate_normals(gl_set.settings.norm_length,gl_set.settings.discretisation)
        normals = utils.apply_transf_2_norms(normals,self._from_mni_to_vox)

        norm_intens = np.array(self._image.interpolate_normals(normals))
        norm_intens = norm_intens.reshape((norm_intens.shape[0]*norm_intens.shape[1]))
        #distr_coords = np.concatenate((coords,norm_intens))

        return - self._kdes(coords,norm_intens)




















class Fitter:

    _pdm = None

    _train_subj = None
    _test_subj = None
    _modal = None
    _mat_over = None
    _best_meshes_mni = None
    _best_meshes_c = None



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
        self._best_meshes_c = []
        for i in range(len(gl_set.settings.labels_to_segment)):


            label = gl_set.settings.labels_to_segment[i]
            # ind = gl_set.settings.all_labels.index(label)
            #
            # sub_ind = list(self._mat_over[ind]).index(1)
            # mni_im = fim.Image(self._train_subj[sub_ind] + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")
            #
            # native_im = fim.Image(self._train_subj[sub_ind] + os.sep + "t1_acpc_extracted.nii.gz")
            #
            # to_mni_mat = fl.readFlirt(self._train_subj[sub_ind] + os.sep + "combined_affine_t1.mat")
            #
            # mni_w = fl.fromFlirt(to_mni_mat, native_im, mni_im, "world", "world")
            #
            # surf_c = ExtPy.cMesh(self._train_subj[sub_ind] + os.sep + label + "_1.obj")
            #surf= mesh.Mesh(self._train_subj[sub_ind] + os.sep + label + "_1.obj")
            surf_c = ExtPy.cMesh(gl_set.settings.atlas_dir + os.sep + label + "m.obj")
            surf = mesh.Mesh(gl_set.settings.atlas_dir + os.sep + label + "m.obj")
            # surf.apply_transform(mni_w)
            # surf_c.applyTransformation(mni_w.tolist())
            self._best_meshes_c.append(surf_c)
            self._best_meshes_mni.append(surf)




    def fit_single(self):
        cds = self._pdm.recompute_conditional_shape_int_distribution(self._best_meshes_mni[0].gen_num_of_points())
        for i in range(len(self._test_subj)):

            for lab in range(len(self._best_meshes_mni)):
                fc = FunctionHandler()
                #todo fix set
                fc.set_image(  self._test_subj[i] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz"   )
                fc.set_subject(self._test_subj[i])
                fc._mesh = self._best_meshes_mni[lab]
                fc._cmesh = self._best_meshes_c[lab]

                fc._mesh.calculate_closest_points()
                fc._kdes = cds[lab]
                fc._num_of_points= self._best_meshes_mni[lab].gen_num_of_points()

                #constraint for interception
                int_Cons = opt.NonlinearConstraint(fc._mesh.calculate_interception_from_newPTS,lb=0,ub=2)


                X0 = fc._mesh.get_unpacked_coords()
                X0 = cds[lab].decompose_coords_to_eigcords(X0)
                X0[:] = 0
                print(datetime.now())
                print(fc._cmesh.selfIntersectionTest(list(cds[lab].vector_2_points(X0))))
                print(datetime.now())
                #print(fc._mesh.calculate_interception_from_newPTS(np.array(X0)))
                print(datetime.now())


                bounds = cds[lab].generate_bounds(3)
                #mimiser = opt.minimize(fc,X0,method="cg",options={"disp":True})

                Xpt = X0.copy()
                for bd in range(1,len(bounds)+1):
                    bound = bounds[-bd]

                    curr = bound[0]
                    arr = []
                    dt = bound[1]*2/100
                    cnt = 0
                    while (bound[0] + dt*cnt < bound[1]):
                        Xpt[-bd] = bound[0] + dt*cnt
                        arr.append(fc(Xpt))
                        cnt = cnt+1

                    arr = np.array(arr)
                    ind = np.where( arr == min(arr))[0][0]
                    Xpt[-bd] = bound[0] +ind*dt

                X0 = Xpt


                print("Start optimising ")
                mimiser = opt.minimize(fc, X0, method='TNC',bounds=bounds, options={"disp": True})
                r_x = mimiser.fun
                for k in range(10):
                    mimiser = opt.minimize(fc, mimiser.x, method='TNC', bounds=bounds, options={"disp": True})
                    # mimiser = opt.minimize(fc, mimiser.x, method='Powell', bounds=bounds,
                    #                       options={"disp": True})

                    if r_x - mimiser.fun < 1:
                        break
                    r_x = mimiser.fun
                #mimiser = opt.minimize(fc, X0, method='L-BFGS-B', bounds=bounds, options={"disp": True})
                #mimiser = opt.minimize(fc, X0, method="cg", options={"disp": True})

                #mimiser = opt.minimize(fc, X0,method='COBYLA',tol=1,constraints=con,options={"maxiter":5000})
                print(datetime.now())
                res_points = cds[lab].vector_2_points(mimiser.x)
                l = len(res_points.tolist())
                fc._mesh.modify_points(res_points.reshape(( int(l/3),3)))
                fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i]))

                fc._mesh.save_obj(self._test_subj[i] + os.sep + str(lab) + "_fitted.obj")

                a = fc(X0)







            pass







        pass